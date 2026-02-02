targetScope = 'resourceGroup'

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('A short token to make resource names unique')
param nameToken string = toLower(uniqueString(resourceGroup().id))

@description('Container Apps Environment name')
param acaEnvName string = 'podcaststudio-${nameToken}'

@description('Container App names')
param backendAppName string = 'podcast-web-${nameToken}'

@description('TTS worker Container App name')
param ttsWorkerAppName string = 'podcast-tts-worker-${nameToken}'

@description('Azure Files share name')
param fileShareName string = 'podcaststudio'

@description('Mount path inside containers')
param storageMountPath string = '/mnt/storage'

@allowed([
  't4'
  'a100'
])
@description('GPU SKU preset for the TTS worker: t4 (cost-efficient) or a100 (best quality)')
param GPU_SKU string = 't4'

@description('Workload profile name for serverless GPU (T4)')
param gpuWorkloadProfileName string = 'gpu'

@description('Workload profile type for serverless GPU (T4).')
param gpuWorkloadProfileType string = 'Consumption-GPU-NC8as-T4'

@description('Workload profile name for serverless GPU (A100)')
param gpuA100WorkloadProfileName string = 'gpu-a100'

@description('Workload profile type for serverless GPU (A100). Must be supported in the chosen region.')
param gpuA100WorkloadProfileType string = 'Consumption-GPU-NC24-A100'

@secure()
@description('Azure OpenAI API key')
param AZURE_OPENAI_API_KEY string

@description('Azure OpenAI endpoint, like https://xxx.openai.azure.com')
param AZURE_OPENAI_ENDPOINT string

@description('Azure OpenAI deployment name')
param AZURE_OPENAI_DEPLOYMENT_NAME string

@description('Optional explicit Azure OpenAI base URL. If empty, backend derives from AZURE_OPENAI_ENDPOINT (adds /openai/v1/).')
param AZURE_OPENAI_BASE_URL string = ''

@description('VibeVoice model id override. Leave empty to derive from GPU_SKU.')
param VIBEVOICE_MODEL_ID string = ''

@description('VibeVoice speaker names comma-separated or quoted string')
param VIBEVOICE_SPEAKER_NAMES string = 'Xinran Anchen'

var resolvedGpuWorkloadProfileType = !empty(gpuWorkloadProfileType)
  ? gpuWorkloadProfileType
  : 'Consumption-GPU-NC8as-T4'

// azd env values are often quoted (e.g. GPU_SKU="a100"). Normalize for comparisons.
var normalizedGpuSku = toLower(trim(replace(GPU_SKU, '"', '')))
var vibeVoiceModelIdOverride = trim(replace(VIBEVOICE_MODEL_ID, '"', ''))

var resolvedTtsWorkerWorkloadProfileName = (normalizedGpuSku == 'a100') ? gpuA100WorkloadProfileName : gpuWorkloadProfileName

var resolvedVibeVoiceModelId = !empty(vibeVoiceModelIdOverride)
  ? vibeVoiceModelIdOverride
  : (normalizedGpuSku == 'a100' ? 'vibevoice/VibeVoice-7B' : 'vibevoice/VibeVoice-1.5B')

@description('Max replicas for all apps')
param maxReplicas int = 1

@description('Min replicas for all apps (set 0 for scale-to-zero)')
param minReplicas int = 0

// During `azd up`, infrastructure is provisioned before service images are pushed to ACR.
// Use public placeholder images so Container Apps can provision successfully, then `azd deploy`
// updates the apps to the correct ACR images.
param placeholderPythonImage string = 'python:3.12-slim'

// Container Apps secret names must be lowercase alphanumeric or '-'.
var openAiSecretName = 'azure-openai-api-key'

var openAiSecret = !empty(AZURE_OPENAI_API_KEY) ? [
  {
    name: openAiSecretName
    value: AZURE_OPENAI_API_KEY
  }
] : []

var openAiEnv = !empty(AZURE_OPENAI_API_KEY) ? [
  {
    name: 'AZURE_OPENAI_API_KEY'
    secretRef: openAiSecretName
  }
] : []

// -----------------------------
// Log Analytics
// -----------------------------
resource la 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'log-${nameToken}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

var laKeys = listKeys(la.id, la.apiVersion)

// -----------------------------
// Storage Account + Azure Files (SMB)
// -----------------------------
// This project mounts Azure Files into Container Apps.
// With shared key access enabled, Container Apps can mount Azure Files via SMB.

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'stn${nameToken}'
  location: location
  tags: {
    SecurityControl: 'Ignore'
    ConstControl: 'Ignore'
  }
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    enableHttpsTrafficOnly: true
    largeFileSharesState: 'Enabled'
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-01-01' = {
  parent: storage
  name: 'default'
}

resource fileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-01-01' = {
  parent: fileService
  name: fileShareName
  properties: {
    shareQuota: 1024
  }
}

var storageKeys = listKeys(storage.id, storage.apiVersion)



// -----------------------------
// Container Apps Environment
// -----------------------------
resource env 'Microsoft.App/managedEnvironments@2025-10-02-preview' = {
  name: acaEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: la.properties.customerId
        sharedKey: laKeys.primarySharedKey
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
      {
        name: gpuWorkloadProfileName
        workloadProfileType: resolvedGpuWorkloadProfileType
      }
      {
        name: gpuA100WorkloadProfileName
        workloadProfileType: gpuA100WorkloadProfileType
      }
    ]
  }
}

// Link Azure Files to the environment
resource envStorage 'Microsoft.App/managedEnvironments/storages@2025-10-02-preview' = {
  name: 'podcaststorage-smb'
  parent: env
  properties: {
    azureFile: {
      accessMode: 'ReadWrite'
      accountName: storage.name
      shareName: fileShare.name
      accountKey: storageKeys.keys[0].value
    }
  }
}

// -----------------------------
// Container Registry
// -----------------------------
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'acr${nameToken}'
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

var acrCreds = listCredentials(acr.id, acr.apiVersion)
var acrUser = acrCreds.username
var acrPass = acrCreds.passwords[0].value

// -----------------------------
// Container Apps
// -----------------------------
var volumeName = 'storage'
var envStorageName = envStorage.name

var commonMounts = [
  {
    volumeName: volumeName
    mountPath: storageMountPath
  }
]

var commonVolumes = [
  {
    name: volumeName
    storageType: 'AzureFile'
    storageName: envStorageName
  }
]

var commonEnv = [
  {
    name: 'PODCASTSTUDIO_STORAGE_DIR'
    value: storageMountPath
  }
  {
    name: 'HF_HOME'
    value: '${storageMountPath}/hf'
  }
  {
    name: 'HUGGINGFACE_HUB_CACHE'
    value: '${storageMountPath}/hf/hub'
  }
  {
    name: 'TRANSFORMERS_CACHE'
    value: '${storageMountPath}/hf/transformers'
  }
]

resource backendApp 'Microsoft.App/containerApps@2025-10-02-preview' = {
  name: backendAppName
  location: location
  properties: {
    environmentId: env.id
    workloadProfileName: 'Consumption'
    configuration: {
      ingress: {
        external: true
        targetPort: 8001
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acrUser
          passwordSecretRef: 'acr-pwd'
        }
      ]
      // Only include the OpenAI key as a secret if it was provided.
      // (Empty secrets are rejected by the Container Apps RP.)
      secrets: concat([
        {
          name: 'acr-pwd'
          value: acrPass
        }
      ], openAiSecret)
    }
    template: {
      containers: [
        {
          name: 'web'
          image: placeholderPythonImage
          resources: {
            cpu: 2
            memory: '4Gi'
          }
          env: concat(commonEnv, openAiEnv, [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: AZURE_OPENAI_ENDPOINT
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
              value: AZURE_OPENAI_DEPLOYMENT_NAME
            }
            {
              name: 'AZURE_OPENAI_BASE_URL'
              value: AZURE_OPENAI_BASE_URL
            }
            {
              name: 'TTS_WORKER_URL'
              value: 'http://${ttsWorkerApp.properties.configuration.ingress.fqdn}'
            }
            {
              name: 'PODCASTSTUDIO_DB_DIR_BACKEND'
              value: '${storageMountPath}/db/backend'
            }
          ])
          volumeMounts: commonMounts
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        // Keep worker warm for 10 minutes after last request.
        cooldownPeriod: 600
        rules: [
          {
            name: 'http'
            http: {
              metadata: {
                // Scale based on concurrent requests; keep maxReplicas=1 so this
                // mainly controls scale-from-zero and cooldown.
                concurrentRequests: '1'
              }
            }
          }
        ]
      }
      volumes: commonVolumes
    }
  }
}

resource ttsWorkerApp 'Microsoft.App/containerApps@2025-10-02-preview' = {
  name: ttsWorkerAppName
  location: location
  properties: {
    environmentId: env.id
    workloadProfileName: resolvedTtsWorkerWorkloadProfileName
    configuration: {
      ingress: {
        external: false
        targetPort: 8002
        transport: 'auto'
        allowInsecure: true
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acrUser
          passwordSecretRef: 'acr-pwd'
        }
      ]
      secrets: concat([
        {
          name: 'acr-pwd'
          value: acrPass
        }
      ], openAiSecret)
    }
    template: {
      containers: [
        {
          name: 'tts-worker'
          image: placeholderPythonImage
          resources: {
            cpu: 8
            gpu: 1
            memory: '56Gi'
          }
          env: concat(commonEnv, openAiEnv, [
            {
              name: 'PODCASTSTUDIO_DB_DIR_TTS_WORKER'
              value: '${storageMountPath}/db/tts_worker'
            }
            {
              name: 'VIBEVOICE_MODEL_ID'
              value: resolvedVibeVoiceModelId
            }
            {
              name: 'VIBEVOICE_SPEAKER_NAMES'
              value: VIBEVOICE_SPEAKER_NAMES
            }
            {
              name: 'VIBEVOICE_REPO_DIR'
              value: '/opt/VibeVoice'
            }
            {
              name: 'VIBEVOICE_ATTN_IMPL'
              value: 'sdpa'
            }
            {
              name: 'VIBEVOICE_LOAD_IN_8BIT'
              value: '0'
            }
            {
              name: 'VIBEVOICE_INPROCESS'
              value: '1'
            }
            {
              name: 'VIBEVOICE_WARMUP'
              value: '1'
            }
            {
              name: 'VIBEVOICE_ALLOW_SUBPROCESS_FALLBACK'
              value: '0'
            }
          ])
          volumeMounts: commonMounts
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
      }
      volumes: commonVolumes
    }
  }
}

output BACKEND_URL string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output BACKEND_APP_NAME string = backendAppName
output TTS_WORKER_APP_NAME string = ttsWorkerAppName
output TTS_WORKER_INTERNAL_URL string = 'http://${ttsWorkerApp.properties.configuration.ingress.fqdn}'
output AZURE_CONTAINER_REGISTRY_NAME string = acr.name
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = acr.properties.loginServer
output STORAGE_ACCOUNT_NAME string = storage.name
output FILE_SHARE_NAME string = fileShare.name
