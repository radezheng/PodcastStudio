const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001'

async function req(path, options = {}) {
  const r = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!r.ok) {
    let detail = ''
    try {
      const j = await r.json()
      detail = j.detail ? `: ${j.detail}` : ''
    } catch {
      try {
        detail = `: ${await r.text()}`
      } catch {
        detail = ''
      }
    }
    throw new Error(`HTTP ${r.status}${detail}`)
  }
  return r
}

export async function generateScript({ theme, minutes }) {
  const r = await req('/api/scripts/generate', {
    method: 'POST',
    body: JSON.stringify({ theme, minutes }),
  })
  return await r.json()
}

export async function getConfig() {
  const r = await req('/api/config')
  return await r.json()
}

export async function saveSystemPrompt(system_prompt) {
  const r = await req('/api/config/system-prompt', {
    method: 'PUT',
    body: JSON.stringify({ system_prompt }),
  })
  return await r.json()
}

export async function generateScriptV2({ theme, minutes, speaker_names, system_prompt }) {
  const r = await req('/api/scripts/generate', {
    method: 'POST',
    body: JSON.stringify({ theme, minutes, speaker_names, system_prompt }),
  })
  return await r.json()
}

export async function generateScriptV3({
  script_id,
  theme,
  minutes,
  speaker_names,
  system_prompt,
  source_filename,
  source_text,
  source_url,
  use_web_search,
}) {
  const r = await req('/api/scripts/generate', {
    method: 'POST',
    body: JSON.stringify({
      script_id,
      theme,
      minutes,
      speaker_names,
      system_prompt,
      source_filename,
      source_text,
      source_url,
      use_web_search: !!use_web_search,
    }),
  })
  return await r.json()
}

export async function getScript(scriptId) {
  const r = await req(`/api/scripts/${scriptId}`)
  return await r.json()
}

export async function listScripts({ limit = 50 } = {}) {
  const r = await req(`/api/scripts?limit=${encodeURIComponent(limit)}`)
  return await r.json()
}

export async function getScriptFull(scriptId) {
  const r = await req(`/api/scripts/${scriptId}/full`)
  return await r.json()
}

export async function getScriptLogs(scriptId) {
  const r = await req(`/api/scripts/${scriptId}/logs`)
  return await r.json()
}

export async function updateScript(scriptId, content) {
  const r = await req(`/api/scripts/${scriptId}`, {
    method: 'PUT',
    body: JSON.stringify({ content }),
  })
  return await r.json()
}

export async function deleteScript(scriptId) {
  const r = await req(`/api/scripts/${scriptId}`, {
    method: 'DELETE',
  })
  return await r.json()
}

export async function confirmScript(scriptId) {
  const r = await req(`/api/scripts/${scriptId}/confirm`, {
    method: 'POST',
  })
  return await r.json()
}

export async function submitTts(scriptId, { speaker_names } = {}) {
  const r = await req(`/api/scripts/${scriptId}/tts`, {
    method: 'POST',
    body: JSON.stringify({ speaker_names }),
  })
  return await r.json()
}

export async function getJob(jobId) {
  const r = await req(`/api/jobs/${jobId}`)
  return await r.json()
}

export function audioUrl(jobId) {
  return `${API_BASE}/api/jobs/${jobId}/audio`
}
