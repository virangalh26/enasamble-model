#!/usr/bin/env bash
# Creates every Azure resource used by the solar detector's MLOps setup.
# Run it in Azure Cloud Shell (Bash) or anywhere `az login` works:
#     LOCATION=<allowed region> bash infra/azure_setup.sh
# It is safe to re-run: existing resources are left as they are.
#
# Resources (all in one resource group, so deleting the group removes everything):
#   Container Registry (Basic)      app images, tagged by git commit
#   Log Analytics workspace         container logs and prediction logs
#   Container Apps environment+app  the public web app (scales to zero)
#   Storage account (+ "dvc" blob)  DVC remote for datasets; Azure ML default storage
#   Azure Machine Learning          MLflow experiment tracking and model registry
#   2 user-assigned identities      id-solar-github: GitHub Actions login (OIDC, no secrets)
#                                   id-solar-app:    lets the app pull images from the registry
set -euo pipefail

: "${LOCATION:?Set LOCATION to a region allowed by your subscription policy}"
REPO="${REPO:-virangalh26/enasamble-model}"
SUB="$(az account show --query id -o tsv)"
TENANT="$(az account show --query tenantId -o tsv)"
SUFFIX="${SUFFIX:-$(printf '%s' "$SUB" | md5sum | cut -c1-6)}"

RG=rg-solar-detector
ACR="acrsolar${SUFFIX}"
ST="stsolar${SUFFIX}"
LAW=log-solar-detector
CAE=cae-solar
APP=ca-solar
AML=mlw-solar-detector
ID_GH=id-solar-github
ID_APP=id-solar-app

az config set extension.use_dynamic_install=yes_without_prompt core.only_show_errors=true >/dev/null

echo "== Registering resource providers (first time only, can take a few minutes)"
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.Storage \
          Microsoft.MachineLearningServices Microsoft.KeyVault Microsoft.Insights Microsoft.ManagedIdentity; do
  az provider register --namespace "$ns" --wait
done

echo "== Resource group, registry, logs"
az group create -n "$RG" -l "$LOCATION" -o none
az acr show -n "$ACR" -o none 2>/dev/null || az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled false -o none
az monitor log-analytics workspace show -g "$RG" -n "$LAW" -o none 2>/dev/null || \
  az monitor log-analytics workspace create -g "$RG" -n "$LAW" -l "$LOCATION" --retention-time 30 -o none

echo "== Storage (DVC remote + Azure ML default storage)"
az storage account show -g "$RG" -n "$ST" -o none 2>/dev/null || \
  az storage account create -g "$RG" -n "$ST" -l "$LOCATION" --sku Standard_LRS --kind StorageV2 \
    --min-tls-version TLS1_2 --allow-blob-public-access false -o none
az storage container-rm create -g "$RG" --storage-account "$ST" -n dvc -o none 2>/dev/null || true

echo "== Managed identities and GitHub OIDC trust"
az identity show -g "$RG" -n "$ID_GH" -o none 2>/dev/null || az identity create -g "$RG" -n "$ID_GH" -o none
az identity show -g "$RG" -n "$ID_APP" -o none 2>/dev/null || az identity create -g "$RG" -n "$ID_APP" -o none
# GitHub may send the subject with names only (repo:owner/repo) or, for newer repositories, with immutable IDs
# (repo:owner@<owner id>/repo@<repo id>), so both forms are trusted. IDs: https://api.github.com/repos/<owner>/<repo>
REPO_IDS="${REPO_IDS:-virangalh26@337202569/enasamble-model@1427262500}"
for pair in "gh-main:repo:${REPO}:ref:refs/heads/main" "gh-production:repo:${REPO}:environment:production" \
            "gh-main-id:repo:${REPO_IDS}:ref:refs/heads/main" "gh-production-id:repo:${REPO_IDS}:environment:production"; do
  name="${pair%%:*}"; subject="${pair#*:}"
  az identity federated-credential show -g "$RG" --identity-name "$ID_GH" -n "$name" -o none 2>/dev/null || \
    az identity federated-credential create -g "$RG" --identity-name "$ID_GH" -n "$name" \
      --issuer https://token.actions.githubusercontent.com --subject "$subject" \
      --audiences api://AzureADTokenExchange -o none
done

echo "== Azure Machine Learning workspace (MLflow tracking + model registry)"
ST_ID="$(az storage account show -g "$RG" -n "$ST" --query id -o tsv)"
az ml workspace show -g "$RG" -n "$AML" -o none 2>/dev/null || \
  az ml workspace create -g "$RG" -n "$AML" -l "$LOCATION" --storage-account "$ST_ID" -o none

echo "== Container Apps environment and app"
if ! az containerapp env show -g "$RG" -n "$CAE" -o none 2>/dev/null; then
  LAW_CID="$(az monitor log-analytics workspace show -g "$RG" -n "$LAW" --query customerId -o tsv)"
  LAW_KEY="$(az monitor log-analytics workspace get-shared-keys -g "$RG" -n "$LAW" --query primarySharedKey -o tsv)"
  # --enable-workload-profiles false: a standard Consumption environment. (An environment created with the
  # defaults came up in "Express" mode, which cannot pull images from a private registry with an identity.)
  az containerapp env create -g "$RG" -n "$CAE" -l "$LOCATION" --enable-workload-profiles false \
    --logs-workspace-id "$LAW_CID" --logs-workspace-key "$LAW_KEY" -o none
fi
APP_ID_RES="$(az identity show -g "$RG" -n "$ID_APP" --query id -o tsv)"
APP_PRINCIPAL="$(az identity show -g "$RG" -n "$ID_APP" --query principalId -o tsv)"
ACR_ID="$(az acr show -n "$ACR" --query id -o tsv)"
az role assignment create --assignee-object-id "$APP_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role AcrPull --scope "$ACR_ID" -o none 2>/dev/null || true
if ! az containerapp show -g "$RG" -n "$APP" -o none 2>/dev/null; then
  # Starts with a placeholder image; the Deploy workflow replaces it with the real app.
  az containerapp create -g "$RG" -n "$APP" --environment "$CAE" \
    --image mcr.microsoft.com/k8se/quickstart:latest --target-port 80 --ingress external \
    --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 \
    --user-assigned "$APP_ID_RES" --registry-server "${ACR}.azurecr.io" --registry-identity "$APP_ID_RES" -o none
fi
# Make sure the app pulls from the registry with its identity (keyless), whatever image it was created with.
az containerapp registry set -g "$RG" -n "$APP" --server "${ACR}.azurecr.io" --identity "$APP_ID_RES" -o none

echo "== Role assignments (least privilege, scoped to this resource group)"
GH_PRINCIPAL="$(az identity show -g "$RG" -n "$ID_GH" --query principalId -o tsv)"
RG_ID="$(az group show -n "$RG" --query id -o tsv)"
AML_ID="$(az ml workspace show -g "$RG" -n "$AML" --query id -o tsv)"
assign() { az role assignment create --assignee-object-id "$1" --assignee-principal-type "$2" --role "$3" --scope "$4" -o none 2>/dev/null || true; }
assign "$GH_PRINCIPAL" ServicePrincipal AcrPush "$ACR_ID"
assign "$GH_PRINCIPAL" ServicePrincipal Contributor "$RG_ID"
assign "$GH_PRINCIPAL" ServicePrincipal "AzureML Data Scientist" "$AML_ID"
assign "$GH_PRINCIPAL" ServicePrincipal "Storage Blob Data Contributor" "$ST_ID"
ME="$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true)"
[ -n "$ME" ] && assign "$ME" User "Storage Blob Data Contributor" "$ST_ID"

echo "== Done. Values for infra/azure.env:"
cat <<EOF
AZURE_SUBSCRIPTION_ID=$SUB
AZURE_TENANT_ID=$TENANT
AZURE_CLIENT_ID=$(az identity show -g "$RG" -n "$ID_GH" --query clientId -o tsv)
LOCATION=$LOCATION
RESOURCE_GROUP=$RG
ACR_NAME=$ACR
CONTAINERAPP_NAME=$APP
AML_WORKSPACE=$AML
STORAGE_ACCOUNT=$ST
MLFLOW_TRACKING_URI=$(az ml workspace show -g "$RG" -n "$AML" --query mlflow_tracking_uri -o tsv)
APP_URL=https://$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)
EOF
