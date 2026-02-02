import React, { useEffect, useMemo, useState } from 'react'
import {
  activateTtsWorker,
  audioUrl,
  confirmScript,
  deleteScript,
  generateScriptV3,
  getConfig,
  getScriptFull,
  getScriptLogs,
  listScripts,
  saveSystemPrompt,
  getJob,
  submitTts,
  updateScript,
} from './api.js'

export default function App() {
  const [theme, setTheme] = useState('边缘AI里的 Agentic Workflow')
  const [minutes, setMinutes] = useState(4)

  const [supportedSpeakers, setSupportedSpeakers] = useState([])
  const [selectedSpeakers, setSelectedSpeakers] = useState(['Xinran', 'Anchen'])
  const [maxSpeakers, setMaxSpeakers] = useState(4)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [builtinSystemPrompt, setBuiltinSystemPrompt] = useState('')

  const [useWebSearch, setUseWebSearch] = useState(true)
  const [sourceFilename, setSourceFilename] = useState('')
  const [sourceText, setSourceText] = useState('')
  const [sourceUrl, setSourceUrl] = useState('')

  const [scriptId, setScriptId] = useState(null)
  const [script, setScript] = useState('')
  const [confirmed, setConfirmed] = useState(false)

  const [jobId, setJobId] = useState(null)
  const [job, setJob] = useState(null)
  const [ttsJobs, setTtsJobs] = useState([])

  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')

  const [logStages, setLogStages] = useState([])

  const [generating, setGenerating] = useState(false)

  const [history, setHistory] = useState([])
  const [selectedHistoryId, setSelectedHistoryId] = useState(null)

  const canSave = useMemo(() => !!scriptId && script.trim().length > 0, [scriptId, script])

  const canGenerate = useMemo(() => {
    return !!theme.trim() || !!sourceText.trim()
  }, [theme, sourceText])

  const jobStatus = job?.worker_status || null
  const shouldPollLogs = useMemo(() => {
    if (!scriptId) return false
    const jobActive = !!jobId && (!jobStatus || (jobStatus !== 'completed' && jobStatus !== 'failed'))
    return generating || jobActive
  }, [scriptId, generating, jobId, jobStatus])

  const preferredDefaultSpeakers = ['Xinran', 'Anchen']
  function pickDefaultSpeakers(list) {
    const supported = Array.isArray(list) ? list.filter(Boolean) : []
    const pickedPreferred = preferredDefaultSpeakers.filter((s) => supported.includes(s))
    if (pickedPreferred.length) return pickedPreferred
    if (supported.length) return supported.slice(0, 2)
    return preferredDefaultSpeakers
  }

  function newUuid() {
    try {
      if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
    } catch {
      // ignore
    }
    // Fallback: not cryptographically strong, but fine for client correlation.
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0
      const v = c === 'x' ? r : (r & 0x3) | 0x8
      return v.toString(16)
    })
  }

  useEffect(() => {
    if (!shouldPollLogs || !scriptId) return
    let alive = true

    const tick = async () => {
      try {
        const logs = await getScriptLogs(scriptId)
        if (!alive) return
        setLogStages(logs.stages || [])
      } catch {
        // non-fatal
      }
    }

    tick()
    const timer = setInterval(tick, 8000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [shouldPollLogs, scriptId])

  useEffect(() => {
    let alive = true
    const boot = async () => {
      // Non-blocking: poke worker early so it can scale-from-zero while UI loads.
      activateTtsWorker()

      try {
        const cfg = await getConfig()
        if (!alive) return
        setSupportedSpeakers(cfg.supported_speakers || [])
        setMaxSpeakers(cfg.max_speakers || 4)
        setSystemPrompt(cfg.default_system_prompt || '')
        setBuiltinSystemPrompt(cfg.builtin_system_prompt || '')
        setSelectedSpeakers(pickDefaultSpeakers(cfg.supported_speakers || []))
      } catch (e) {
        // Non-fatal; user can still use the app with defaults.
      }

      try {
        const h = await listScripts({ limit: 80 })
        if (!alive) return
        setHistory(h.scripts || [])
      } catch (e) {
        // non-fatal
      }
    }
    boot()
    return () => {
      alive = false
    }
  }, [])

  async function refreshHistory({ selectScriptId } = {}) {
    try {
      const h = await listScripts({ limit: 80 })
      setHistory(h.scripts || [])
      if (selectScriptId) {
        setSelectedHistoryId(selectScriptId)
      }
    } catch {
      // non-fatal
    }
  }

  function onNew() {
    setSelectedHistoryId(null)
    setScriptId(null)
    setScript('')
    setConfirmed(false)
    setJobId(null)
    setJob(null)
    setTtsJobs([])
    setLogStages([])
    setSourceFilename('')
    setSourceText('')
    setSourceUrl('')
    setUseWebSearch(true)
    setTheme('')
    setMinutes(4)
    setSelectedSpeakers(pickDefaultSpeakers(supportedSpeakers || []))
    setErr('')
    setMsg('New draft.')
  }

  async function loadHistoryItem(id) {
    if (!id) return
    setBusy(true)
    setErr('')
    setMsg('Loading history...')
    setSelectedHistoryId(id)
    setLogStages([])
    try {
      const full = await getScriptFull(id)

      const s = full.script
      setScriptId(s.script_id)
      setTheme(s.theme || '')
      setScript(s.content || '')
      setConfirmed(!!s.confirmed)

      const meta = full.meta || null
      if (meta) {
        setMinutes(meta.minutes || 4)
        setUseWebSearch(!!meta.use_web_search)
        setSourceFilename(meta.source_filename || '')
        setSourceUrl(meta.source_url || '')
        if (meta.system_prompt) setSystemPrompt(meta.system_prompt)
        if (Array.isArray(meta.speaker_names) && meta.speaker_names.length) {
          setSelectedSpeakers(meta.speaker_names)
        }
      }

      setTtsJobs(full.tts_jobs || [])
      setJobId(null)
      setJob(null)

      try {
        const logs = await getScriptLogs(id)
        setLogStages(logs.stages || [])
      } catch {
        // non-fatal
      }

      setMsg('History loaded.')
    } catch (e) {
      setErr(e.message)
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

  async function onDeleteScript(id) {
    if (!id) return
    const ok = window.confirm('Delete this script from history? This will also remove its TTS jobs from the backend DB.')
    if (!ok) return

    setBusy(true)
    setErr('')
    setMsg('Deleting...')
    try {
      await deleteScript(id)
      await refreshHistory({ selectScriptId: null })

      if (scriptId === id) {
        setScriptId(null)
        setScript('')
        setConfirmed(false)
        setJobId(null)
        setJob(null)
        setTtsJobs([])
        setLogStages([])
      }
      setSelectedHistoryId(null)
      setMsg('Deleted.')
    } catch (e) {
      setErr(e.message)
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

  function toggleSpeaker(name) {
    setSelectedSpeakers((prev) => {
      const has = prev.includes(name)
      if (has) return prev.filter((x) => x !== name)
      if (prev.length >= maxSpeakers) return prev
      return [...prev, name]
    })
  }

  async function onGenerate() {
    setBusy(true)
    setGenerating(true)
    setErr('')
    setMsg('Generating script...')
    setJobId(null)
    setJob(null)
    setTtsJobs([])
    setLogStages([])

    const clientScriptId = newUuid()
    setScriptId(clientScriptId)
    setSelectedHistoryId(clientScriptId)
    try {
      const r = await generateScriptV3({
        script_id: clientScriptId,
        theme,
        minutes,
        speaker_names: selectedSpeakers,
        system_prompt: systemPrompt,
        source_filename: sourceFilename || undefined,
        source_text: sourceText || undefined,
        source_url: sourceUrl || undefined,
        use_web_search: useWebSearch,
      })
      setScriptId(r.script_id)
      setScript(r.content)
      setConfirmed(r.confirmed)
      setSelectedHistoryId(r.script_id)
      setMsg('Script generated. Review & edit, then confirm.')

      await refreshHistory({ selectScriptId: r.script_id })

      try {
        const logs = await getScriptLogs(r.script_id)
        setLogStages(logs.stages || [])
      } catch (e) {
        // non-fatal
      }
    } catch (e) {
      setErr(e.message)
      setMsg('')
    } finally {
      setGenerating(false)
      setBusy(false)
    }
  }

  async function onPickFile(file) {
    if (!file) return
    const name = file.name || ''
    setSourceFilename(name)
    setErr('')
    setMsg('Reading file...')
    try {
      const text = await file.text()
      setSourceText(text || '')
      setMsg(`Loaded file: ${name}`)
    } catch (e) {
      setErr(e.message)
      setMsg('')
    }
  }

  async function onSaveSystemPrompt() {
    setBusy(true)
    setErr('')
    setMsg('Saving system prompt...')
    try {
      const cfg = await saveSystemPrompt(systemPrompt)
      setSupportedSpeakers(cfg.supported_speakers || [])
      setMaxSpeakers(cfg.max_speakers || 4)
      setSystemPrompt(cfg.default_system_prompt || systemPrompt)
      setBuiltinSystemPrompt(cfg.builtin_system_prompt || builtinSystemPrompt)
      setMsg('System prompt saved.')
    } catch (e) {
      setErr(e.message)
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

  function onUseDefaultSystemPrompt() {
    if (!builtinSystemPrompt) return
    setSystemPrompt(builtinSystemPrompt)
    setMsg('Reset to default prompt. Click Save to persist.')
    setErr('')
  }

  async function onSave() {
    if (!canSave) return
    setBusy(true)
    setErr('')
    setMsg('Saving script...')
    try {
      const r = await updateScript(scriptId, script)
      setConfirmed(r.confirmed)
      setMsg('Saved.')
    } catch (e) {
      setErr(e.message)
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

  async function onConfirm() {
    if (!scriptId) return
    setBusy(true)
    setErr('')
    setMsg('Confirming...')
    try {
      await onSave()
      const r = await confirmScript(scriptId)
      setConfirmed(r.confirmed)
      setMsg('Confirmed. You can generate audio now.')
    } catch (e) {
      setErr(e.message)
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

  async function onGenerateAudio() {
    if (!scriptId) return
    setBusy(true)
    setErr('')
    setMsg('Submitting TTS job...')
    try {
      const r = await submitTts(scriptId, { speaker_names: selectedSpeakers })
      setJobId(r.job_id)
      setMsg('Job submitted. Waiting for completion...')

      // Refresh outputs list for this script
      try {
        const full = await getScriptFull(scriptId)
        setTtsJobs(full.tts_jobs || [])
        await refreshHistory({ selectScriptId: scriptId })
      } catch {
        // non-fatal
      }
    } catch (e) {
      setErr(e.message)
      setMsg('')
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    if (!jobId) return
    let alive = true
    const tick = async () => {
      try {
        const j = await getJob(jobId)
        if (!alive) return
        setJob(j)

        if (j.worker_status) {
          setTtsJobs((prev) =>
            (prev || []).map((it) =>
              it.job_id === jobId ? { ...it, worker_status: j.worker_status, status: j.worker_status } : it
            )
          )
        }

        if (j.worker_status === 'completed' || j.worker_status === 'failed') {
          try {
            const full = await getScriptFull(j.script_id)
            if (!alive) return
            setTtsJobs(full.tts_jobs || [])
            await refreshHistory({ selectScriptId: j.script_id })
          } catch {
            // non-fatal
          }
          return
        }
        setTimeout(tick, 1500)
      } catch (e) {
        if (!alive) return
        setErr(e.message)
      }
    }
    tick()
    return () => {
      alive = false
    }
  }, [jobId])

  return (
    <div className="container">
      <h1>VibeVoice Podcast Studio</h1>
      <div className="appShell">
        <div className="sidebar">
          <div className="card" style={{ marginTop: 0 }}>
            <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="small">History</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={() => refreshHistory({ selectScriptId: selectedHistoryId })}
                  disabled={busy}
                  style={{ padding: '6px 10px' }}
                >
                  Refresh
                </button>
                <button onClick={onNew} disabled={busy} style={{ padding: '6px 10px' }}>
                  New
                </button>
              </div>
            </div>

            {history.length === 0 ? (
              <div className="small" style={{ marginTop: 10, opacity: 0.8 }}>
                No scripts yet.
              </div>
            ) : (
              <div style={{ marginTop: 10 }}>
                {history.map((item) => {
                  const active = item.script_id === selectedHistoryId
                  const subtitle = `${item.confirmed ? 'confirmed' : 'draft'} • ${item.job_count || 0} audio job(s)`
                  return (
                    <div
                      key={item.script_id}
                      className={`historyItem ${active ? 'active' : ''}`}
                      onClick={() => loadHistoryItem(item.script_id)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') loadHistoryItem(item.script_id)
                      }}
                      title={item.theme}
                    >
                      <div className="historyRow">
                        <div className="historyTitle">{item.theme || '(untitled)'}</div>
                        {active && (
                          <button
                            className="danger"
                            onClick={(e) => {
                              e.preventDefault()
                              e.stopPropagation()
                              onDeleteScript(item.script_id)
                            }}
                            disabled={busy}
                            style={{ padding: '6px 10px' }}
                            title="Delete"
                          >
                            Del
                          </button>
                        )}
                      </div>
                      <div className="small" style={{ opacity: 0.75, marginTop: 4 }}>
                        {subtitle}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        <div className="main">
          <div className="card">
        <div className="row">
          <div style={{ flex: 2, minWidth: 280 }}>
            <label>Topic / Theme / Link</label>
            <input value={theme} onChange={(e) => setTheme(e.target.value)} />
            <div className="small" style={{ marginTop: 6, opacity: 0.8 }}>
              Optional link: if valid, backend will download it as source material.
            </div>
            <input
              value={sourceUrl}
              onChange={(e) => setSourceUrl(e.target.value)}
              placeholder="https://... (.txt/.md or text content)"
              disabled={busy}
              style={{ marginTop: 8 }}
            />
          </div>
          <div style={{ width: 160 }}>
            <label>Minutes</label>
            <input
              type="number"
              min={1}
              max={20}
              value={minutes}
              onChange={(e) => setMinutes(parseInt(e.target.value || '4', 10))}
            />
          </div>
          <div style={{ alignSelf: 'flex-end' }}>
            <button className="primary" onClick={onGenerate} disabled={busy || !canGenerate}>
              Generate Script
            </button>
          </div>
        </div>

        <div className="row" style={{ marginTop: 12, alignItems: 'flex-end' }}>
          <div style={{ flex: 2, minWidth: 280 }}>
            <label>Optional source file (.txt / .md)</label>
            <input
              type="file"
              accept=".txt,.md,text/plain,text/markdown"
              onChange={(e) => onPickFile(e.target.files?.[0])}
              disabled={busy}
            />
            {sourceFilename && (
              <div className="small" style={{ marginTop: 6, opacity: 0.8 }}>
                Using: {sourceFilename}
              </div>
            )}
          </div>
          <div style={{ width: 220 }}>
            <label>Web search</label>
            <label className="small" style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
              <input
                type="checkbox"
                checked={useWebSearch}
                onChange={(e) => setUseWebSearch(e.target.checked)}
                disabled={busy}
              />
              Use web search for more materials
            </label>
          </div>
        </div>

        <div className="row" style={{ marginTop: 12, alignItems: 'flex-start' }}>
          <div style={{ flex: 1, minWidth: 260 }}>
            <label>
              Speakers (select up to {maxSpeakers})
              <span className="small" style={{ marginLeft: 8, opacity: 0.8 }}>
                Selected: {selectedSpeakers.length}
              </span>
            </label>
            <div className="row" style={{ gap: 10, flexWrap: 'wrap', justifyContent: 'flex-start' }}>
              {(supportedSpeakers.length ? supportedSpeakers : ['Xinran', 'Anchen']).map((name) => {
                const checked = selectedSpeakers.includes(name)
                const disabled = !checked && selectedSpeakers.length >= maxSpeakers
                return (
                  <label key={name} className="small" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={disabled}
                      onChange={() => toggleSpeaker(name)}
                    />
                    {name}
                  </label>
                )
              })}
            </div>
          </div>

          <div style={{ flex: 2, minWidth: 320 }}>
            <label>System prompt (editable)</label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Enter system prompt used for script generation."
              style={{ minHeight: 90 }}
            />

            <div className="row" style={{ marginTop: 8, justifyContent: 'flex-end' }}>
              <button onClick={onSaveSystemPrompt} disabled={busy || !systemPrompt.trim()}>
                Save System Prompt
              </button>
              <button onClick={onUseDefaultSystemPrompt} disabled={busy || !builtinSystemPrompt.trim()}>
                Use default
              </button>
            </div>
          </div>
        </div>

        {msg && <div className="small success" style={{ marginTop: 10 }}>{msg}</div>}
        {err && <div className="small error" style={{ marginTop: 10 }}>{err}</div>}
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="small">Script ID: {scriptId || '-'}</div>
          <div className="small">Confirmed: {confirmed ? 'yes' : 'no'}</div>
        </div>
        <label style={{ marginTop: 10 }}>Script (editable)</label>
        <textarea value={script} onChange={(e) => setScript(e.target.value)} placeholder="Generate a script to begin." />
        <div className="row" style={{ marginTop: 10 }}>
          <button onClick={onSave} disabled={busy || !canSave}>Save</button>
          <button className="primary" onClick={onConfirm} disabled={busy || !canSave}>Confirm</button>
          <button onClick={onGenerateAudio} disabled={busy || !confirmed}>Generate Audio</button>
        </div>
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="small">Job ID: {jobId || '-'}</div>
          <div className="small">Worker status: {job?.worker_status || '-'}</div>
        </div>

        {ttsJobs.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div className="small" style={{ opacity: 0.8, marginBottom: 6 }}>Outputs</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {ttsJobs.map((j) => (
                <div
                  key={j.job_id}
                  className={`historyItem ${j.job_id === jobId ? 'active' : ''}`}
                  onClick={() => setJobId(j.job_id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') setJobId(j.job_id)
                  }}
                >
                  <div className="historyTitle">Job: {j.job_id}</div>
                  <div className="small" style={{ opacity: 0.75, marginTop: 4 }}>
                    status: {j.worker_status || j.status}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {job?.worker_status === 'completed' && (
          <div style={{ marginTop: 12 }}>
            <audio controls src={audioUrl(jobId)} style={{ width: '100%' }} />
            <div className="small" style={{ marginTop: 8 }}>
              <a href={audioUrl(jobId)} target="_blank" rel="noreferrer">Download wav</a>
            </div>
          </div>
        )}
        {job?.worker_status === 'failed' && (
          <div className="small error" style={{ marginTop: 12 }}>
            Worker error: {job?.error || 'unknown'}
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="small">Backend logs (staged)</div>
          <div className="small" style={{ opacity: 0.8 }}>
            {scriptId ? `Script: ${scriptId}` : '-'}
          </div>
        </div>

        {logStages.length === 0 ? (
          <div className="small" style={{ marginTop: 10, opacity: 0.8 }}>
            No logs yet. Generate a script to see step-by-step backend logs.
          </div>
        ) : (
          <div style={{ marginTop: 10, maxHeight: 260, overflow: 'auto' }}>
            {logStages.map((stage) => (
              <details key={stage.stage} open>
                <summary className="small" style={{ cursor: 'pointer' }}>
                  {stage.title || stage.stage}
                </summary>
                <div className="small" style={{ marginTop: 8, whiteSpace: 'pre-wrap' }}>
                  {(stage.events || [])
                    .filter((e) => e.type !== 'stage_start' && e.type !== 'stage_end')
                    .map((e, idx) => {
                      const extra = e.data ? ` ${JSON.stringify(e.data)}` : ''
                      return `${idx + 1}. ${e.message}${extra}`
                    })
                    .join('\n')}
                </div>
              </details>
            ))}
          </div>
        )}
      </div>
        </div>
      </div>
    </div>
  )
}
