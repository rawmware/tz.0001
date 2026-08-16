import './styles.css';
import { CreateMLCEngine, CreateWebWorkerMLCEngine, prebuiltAppConfig, type MLCEngineInterface } from '@mlc-ai/web-llm';

type Role = 'user' | 'assistant';
type Msg = { role: Role; content: string; turn?: number; pending?: boolean };
type Route = 'Instant' | 'On-device' | 'External';
type Turn = { id: number; route: Route; model: string; input: number; output: number; ms: number; tps: number; ttft: number };
type ApiConfig = { endpoint: string; model: string; key: string };
type GPUAdapterLike = { info?: { vendor?: string; architecture?: string; device?: string; description?: string }; features?: Set<string>; limits?: Record<string, number> };
type ModelRecordLike = (typeof prebuiltAppConfig.model_list)[number];
type CpuChatMessage = { role: string; content: string };
type CpuGenerator = ((input: CpuChatMessage[], options?: Record<string, unknown>) => Promise<unknown>) & { tokenizer: any; dispose?: () => Promise<void> };

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const status = $<HTMLParagraphElement>('status');
const prompt = $<HTMLTextAreaElement>('prompt');
const sendBtn = $<HTMLButtonElement>('sendBtn');
const form = $<HTMLFormElement>('chatForm');
const messagesEl = $<HTMLDivElement>('messages');
const errorEl = $<HTMLParagraphElement>('error');
const ledger = $<HTMLTableSectionElement>('ledger');
const hardware = $<HTMLElement>('hardware');
const runtimeDetail = $<HTMLElement>('modelAudit');
const connectBtn = $<HTMLButtonElement>('connectBtn');
const connectPanel = $<HTMLElement>('connectPanel');
const apiForm = $<HTMLFormElement>('apiForm');
const apiEndpoint = $<HTMLInputElement>('apiEndpoint');
const apiModel = $<HTMLInputElement>('apiModel');
const apiKey = $<HTMLInputElement>('apiKey');
const apiStatus = $<HTMLElement>('apiStatus');
const disconnectApiBtn = $<HTMLButtonElement>('disconnectApiBtn');

const escapeHtml = (value: string) => value.replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[char] || char));
const estimateTokens = (value: string) => Math.max(1, Math.ceil(value.length / 4));
const normalizeOutput = (value: string) => value.trim().replace(/^```[\w-]*\s*/, '').replace(/\s*```$/, '');
const tzSystemPrompt = 'You are TZ, a fast and capable general-purpose assistant running on the user\'s device. Answer immediately and follow the request precisely. For writing or code, return the finished usable content unless explanation is requested. For code, prefer complete runnable code over pseudocode. For factual answers, be concise and state uncertainty instead of inventing facts. Never mention another assistant or provider unless relevant. Do not output Markdown triple-backtick fences because the TZ interface already puts every answer in a copyable code box.'; const cpuSystemPrompt = 'You are TZ. Answer briefly, directly, and accurately. Use plain text. For code, return usable code. Do not discuss the runtime unless asked.';

let engine: MLCEngineInterface | null = null;
let enginePromise: Promise<void> | null = null;
let engineWorker: Worker | null = null;
let history: Msg[] = [];
let turns: Turn[] = [];
let runtimeError = '';
let apiConfig: ApiConfig | null = null;
let adapter: GPUAdapterLike | null = null;
let adapterFeatures = new Set<string>();
let selectedModelRecord: ModelRecordLike | null = null;
let modelCandidates: ModelRecordLike[] = [];
let modelSelectionReason = 'Waiting for hardware';
let modelBudgetMB = 0;
let engineModelId = '';
let lastRoute: Route = 'Instant';
let loadProgress = 0;
let loadText = 'Waiting for hardware';
let loadStarted = 0;
let loadMs = 0;
let localTps = 0;
let localTtft = 0;
let localRuntimeStats = 'No local generation yet';
let localThread = 'Not started';
let storageUsage = 0;
let storageQuota = 0;
let storagePersistent = false;
const PRIMARY_GPU_MODEL_IDS = ['Qwen2.5-0.5B-Instruct-q4f16_1-MLC', 'Qwen2.5-0.5B-Instruct-q4f32_1-MLC', 'SmolLM2-360M-Instruct-q4f16_1-MLC', 'SmolLM2-360M-Instruct-q4f32_1-MLC'];
const FAST_CPU_MODEL_ID = 'onnx-community/SmolLM2-135M-Instruct-ONNX';
let fastModelId = FAST_CPU_MODEL_ID;
let cpuGenerator: CpuGenerator | null = null;
let cpuPromise: Promise<void> | null = null;
let cpuLoadProgress = 0;
let cpuLoadText = 'Fast local idle';
let cpuLoadMs = 0;
let measuredLocalBackend = '';
let cpuDeviceLabel = 'CPU / WebAssembly';
let cpuDtype = 'q4';
let localLane = 'Not run yet';
let webnnAvailable = false;
let wasmThreads = 1;

const calculateLocal = (input: string): string | null => {
    const normalized = input.trim().replace(/^(?:what is|what\'s|calculate|compute|solve)\s+/i, '').replace(/[×x]/gi, '*').replace(/÷/g, '/').replace(/\?$/, '').trim();
    if (!normalized || !/^[0-9+\-*/().%\s]+$/.test(normalized)) return null;
    const tokens = normalized.match(/\d+(?:\.\d+)?|[()+\-*/%]/g);
    if (!tokens) return null;
    let i = 0;
    const expr = (): number => {
        let value = term();
        while (tokens[i] === '+' || tokens[i] === '-') {
            const op = tokens[i++];
            const right = term();
            value = op === '+' ? value + right : value - right;
        }
        return value;
    };
    const term = (): number => {
        let value = factor();
        while (tokens[i] === '*' || tokens[i] === '/' || tokens[i] === '%') {
            const op = tokens[i++];
            const right = factor();
            if ((op === '/' || op === '%') && right === 0) throw new Error('zero');
            value = op === '*' ? value * right : op === '/' ? value / right : value % right;
        }
        return value;
    };
    const factor = (): number => {
        if (tokens[i] === '+') { i++; return factor(); }
        if (tokens[i] === '-') { i++; return -factor(); }
        if (tokens[i] === '(') { i++; const value = expr(); if (tokens[i++] !== ')') throw new Error('paren'); return value; }
        const value = Number(tokens[i++]);
        if (!Number.isFinite(value)) throw new Error('number');
        return value;
    };
    try {
        const result = expr();
        if (i !== tokens.length || !Number.isFinite(result)) return null;
        return Number.isInteger(result) ? String(result) : String(Number(result.toPrecision(12)));
    } catch {
        return null;
    }
};

const fullWidth = (value: string) => [...value.toUpperCase()].map(char => {
    const code = char.charCodeAt(0);
    return code >= 33 && code <= 126 ? String.fromCharCode(code + 65248) : char;
}).join('');
const smallCaps = (value: string) => {
    const map: Record<string, string> = { a: 'ᴀ', b: 'ʙ', c: 'ᴄ', d: 'ᴅ', e: 'ᴇ', f: 'ꜰ', g: 'ɢ', h: 'ʜ', i: 'ɪ', j: 'ᴊ', k: 'ᴋ', l: 'ʟ', m: 'ᴍ', n: 'ɴ', o: 'ᴏ', p: 'ᴘ', q: 'Q', r: 'ʀ', s: 'ꜱ', t: 'ᴛ', u: 'ᴜ', v: 'ᴠ', w: 'ᴡ', x: 'x', y: 'ʏ', z: 'ᴢ' };
    return [...value.toLowerCase()].map(char => map[char] || char).join('');
};
const instantTool = (input: string): string | null => {
    const calc = calculateLocal(input);
    if (calc !== null) return calc;
    const text = input.trim();
    if (/^test[.!?]*$/i.test(text)) return 'TZ is online and ready.';
    const style = text.match(/^write\s+(?:the\s+name\s+)?(.+?)\s+and\s+make\s+it\s+look\s+(?:cool|stylish|fancy)[.!?]*$/i);
    if (style) {
        const value = style[1].replace(/^['"]|['"]$/g, '').trim();
        const upper = value.toUpperCase();
        return `${upper}\n${fullWidth(value)}\n${smallCaps(value)}\n${[...upper].join(' ')}\n✦ ${upper} ✦`;
    }
    return null;
};

const copyText = async (value: string) => {
    try {
        await navigator.clipboard.writeText(value);
    } catch {
        const area = document.createElement('textarea');
        area.value = value;
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        document.execCommand('copy');
        area.remove();
    }
};

const renderMessages = () => {
    messagesEl.innerHTML = '';
    if (!history.length) {
        messagesEl.innerHTML = '<div class="empty">Ask TZ anything.</div>';
        return;
    }
    history.forEach(message => {
        const div = document.createElement('div');
        div.className = `message ${message.role}`;
        if (message.role === 'user') {
            div.textContent = message.content;
        } else if (message.pending && !message.content) {
            const thinking = document.createElement('div');
            thinking.className = 'thinking';
            thinking.textContent = 'TZ is working locally…';
            div.appendChild(thinking);
        } else {
            const card = document.createElement('div');
            card.className = 'answer-card';
            const toolbar = document.createElement('div');
            toolbar.className = 'answer-toolbar';
            const label = document.createElement('span');
            label.textContent = 'TZ';
            const copy = document.createElement('button');
            copy.type = 'button';
            copy.className = 'answer-copy';
            copy.textContent = 'Copy';
            copy.setAttribute('aria-label', 'Copy TZ answer');
            copy.addEventListener('click', async () => {
                await copyText(message.content);
                copy.textContent = 'Copied';
                setTimeout(() => copy.textContent = 'Copy', 1200);
            });
            toolbar.append(label, copy);
            const pre = document.createElement('pre');
            pre.className = 'answer-code';
            const code = document.createElement('code');
            code.textContent = message.content;
            pre.appendChild(code);
            card.append(toolbar, pre);
            div.appendChild(card);
        }
        if (message.role === 'assistant' && message.turn && !message.pending) {
            const turn = turns.find(item => item.id === message.turn);
            if (turn) {
                const meta = document.createElement('div');
                meta.className = 'turn-meta';
                const speed = turn.tps > 0 ? ` · ${turn.tps.toFixed(1)} tok/s` : '';
                meta.textContent = `${turn.output} output tokens${speed} · ${turn.route}`;
                div.appendChild(meta);
            }
        }
        messagesEl.appendChild(div);
    });
    messagesEl.scrollTop = messagesEl.scrollHeight;
};

const renderTelemetry = () => {
    const total = turns.reduce((sum, turn) => sum + turn.input + turn.output, 0);
    const last = turns.at(-1);
    $<HTMLElement>('mTokens').textContent = `${total.toLocaleString()} session tokens`;
    $<HTMLElement>('mLocalSpeed').textContent = localTps > 0 ? `${localTps.toFixed(1)}` : '—';
    $<HTMLElement>('mLatency').textContent = last ? `${(last.ms / 1000).toFixed(2)}s` : '—';
    $<HTMLElement>('mFirstToken').textContent = last && last.ttft > 0 ? `first token ${(last.ttft / 1000).toFixed(2)}s` : 'first token —';
    $<HTMLElement>('mGpu').textContent = engine ? 'WEBGPU' : cpuGenerator ? 'WASM' : adapter ? 'WEBGPU' : 'OFF';
    $<HTMLElement>('mLoad').textContent = engine || cpuGenerator ? '100%' : enginePromise ? `${loadProgress}%` : cpuPromise ? `${cpuLoadProgress}%` : runtimeError ? 'RETRY' : '0%';
    $<HTMLElement>('mLoadText').textContent = engine ? `WebGPU loaded in ${(loadMs / 1000).toFixed(1)}s` : cpuGenerator ? `${cpuDeviceLabel} loaded in ${(cpuLoadMs / 1000).toFixed(1)}s` : cpuPromise ? cpuLoadText.slice(0, 54) : loadText.slice(0, 54);
    ledger.innerHTML = turns.length ? turns.map(turn => `<tr><td>${turn.id}</td><td>TZ</td><td>${turn.input}</td><td>${turn.output}</td><td>${turn.ms ? `${(turn.ms / 1000).toFixed(2)}s` : 'instant'}</td><td>${turn.tps > 0 ? turn.tps.toFixed(1) : '—'}</td><td>${escapeHtml(turn.route)}</td></tr>`).join('') : '<tr><td colspan="7">No turns yet.</td></tr>';
    $<HTMLElement>('promptRouteAudit').textContent = lastRoute;
};

const detectBrowser = () => {
    const ua = navigator.userAgent;
    const match = ua.match(/(?:CriOS|Chrome)\/(\d+)/) || ua.match(/Version\/(\d+).*Safari/) || ua.match(/FxiOS\/(\d+)/) || ua.match(/Edg\/(\d+)/);
    const version = match?.[1] ? ` ${match[1]}` : '';
    if (/Edg\//.test(ua)) return `Microsoft Edge${version}`;
    if (/CriOS|Chrome\//.test(ua)) return `Chrome${version}`;
    if (/FxiOS|Firefox\//.test(ua)) return `Firefox${version}`;
    if (/Safari\//.test(ua)) return `Safari${version}`;
    return 'Browser';
};
const detectDeviceFamily = () => {
    const ua = navigator.userAgent;
    if (/iPhone/i.test(ua)) return 'Apple iPhone';
    if (/iPad/i.test(ua) || navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1) return 'Apple iPad';
    if (/Android/i.test(ua)) return 'Android device';
    if (/Mac/i.test(navigator.platform)) return 'Apple Mac';
    if (/Win/i.test(navigator.platform)) return 'Windows PC';
    if (/Linux/i.test(navigator.platform)) return 'Linux device';
    return 'Current device';
};
const isMobileDevice = () => /iPhone|iPad|Android/i.test(navigator.userAgent) || navigator.maxTouchPoints > 1 && Math.min(screen.width, screen.height) < 900;
const displayProfile = () => `${Math.min(screen.width, screen.height)}×${Math.max(screen.width, screen.height)} CSS px @ ${devicePixelRatio.toFixed(1)}x`;
const highEntropyHardware = async () => {
    const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { getHighEntropyValues?: (keys: string[]) => Promise<Record<string, unknown>> } };
    try {
        return await nav.userAgentData?.getHighEntropyValues?.(['architecture', 'bitness', 'model']) || {};
    } catch {
        return {};
    }
};
const modelSizeB = (modelId: string) => {
    const matches = [...modelId.matchAll(/(\d+(?:\.\d+)?)B/gi)];
    return matches.length ? Number(matches[matches.length - 1][1]) : 0.2;
};
const modelMemoryMB = (record: ModelRecordLike) => Number(record.vram_required_MB || 0);
const modelIsCompatible = (record: ModelRecordLike, limits: Record<string, number>) => {
    const required = Array.isArray(record.required_features) ? record.required_features.map(String) : [];
    if (!required.every(feature => adapterFeatures.has(feature))) return false;
    const storageLimit = Number(limits.maxStorageBufferBindingSize || 0);
    const requiredBuffer = Number(record.buffer_size_required_bytes || 0);
    if (storageLimit && requiredBuffer && requiredBuffer > storageLimit) return false;
    return !/vision/i.test(record.model_id);
};
const planModels = () => {
    const limits = adapter?.limits || {};
    const compatible = prebuiltAppConfig.model_list.filter(record => modelIsCompatible(record, limits));
    const preferred = PRIMARY_GPU_MODEL_IDS.map(modelId => compatible.find(record => record.model_id === modelId)).filter((record): record is ModelRecordLike => Boolean(record));
    const knownFallbacks = compatible.filter(record => /Qwen2\.5-0\.5B-Instruct|SmolLM2-360M-Instruct/i.test(record.model_id)).sort((a, b) => modelMemoryMB(a) - modelMemoryMB(b));
    modelCandidates = [...preferred, ...knownFallbacks].filter((record, index, all) => all.findIndex(item => item.model_id === record.model_id) === index).slice(0, 4);
    selectedModelRecord = modelCandidates[0] || null;
    modelBudgetMB = selectedModelRecord ? Math.ceil(modelMemoryMB(selectedModelRecord)) : 0;
    const precision = selectedModelRecord?.model_id.includes('q4f16') ? '4-bit weights + f16 shaders' : selectedModelRecord?.model_id.includes('q4f32') ? '4-bit weights + f32 shaders' : 'compatible WebGPU build';
    modelSelectionReason = selectedModelRecord ? `production allowlist · ${precision} · one model loaded at a time` : 'No supported WebLLM model was found in this runtime build';
};

const refreshStorage = async () => {
    try {
        storagePersistent = await navigator.storage.persisted();
        const estimate = await navigator.storage.estimate();
        storageUsage = estimate.usage || 0;
        storageQuota = estimate.quota || 0;
    } catch {
        storagePersistent = false;
    }
};

const renderRuntime = () => {
    const record = selectedModelRecord;
    const required = record && Array.isArray(record.required_features) ? record.required_features.map(String) : [];
    const compatible = Boolean(adapter) && required.every(feature => adapterFeatures.has(feature));
    const localReady = Boolean(engine || cpuGenerator);
    const state = engine ? 'WebGPU model resident and ready' : cpuGenerator ? 'CPU/WASM emergency model resident and ready' : enginePromise ? 'Loading the WebGPU model' : cpuPromise ? 'Loading the CPU/WASM emergency model' : runtimeError ? 'Local runtime needs a retry' : adapter ? 'WebGPU model queued' : 'WebGPU unavailable · CPU/WASM emergency model queued';
    const plannedBackend = adapter ? 'WebGPU / WebLLM → CPU/WASM only after a real WebGPU failure' : 'CPU/WASM compact emergency path';
    status.textContent = localReady ? 'TZ ready · running locally' : enginePromise ? `Preparing TZ · ${loadProgress}%` : cpuPromise ? `Preparing fallback · ${cpuLoadProgress}%` : runtimeError ? 'TZ local engine needs a retry' : 'Preparing TZ';
    $<HTMLElement>('activeModelAudit').textContent = engine ? engineModelId : cpuGenerator ? fastModelId : record?.model_id || fastModelId;
    $<HTMLElement>('localStatus').textContent = engine ? 'Local · WebGPU ready' : cpuGenerator ? 'Local · CPU/WASM fallback ready' : enginePromise ? `WebGPU loading ${loadProgress}%` : cpuPromise ? `CPU/WASM loading ${cpuLoadProgress}%` : runtimeError ? apiConfig ? 'Local retry needed · external fallback ready' : 'Local runtime retry needed' : adapter ? 'WebGPU detected · preparing' : 'WebGPU unavailable · preparing CPU/WASM';
    $<HTMLElement>('gpuAudit').textContent = engine ? 'WebGPU active on this device' : adapter ? 'WebGPU detected · model preparing' : cpuGenerator ? 'WebGPU unavailable · CPU/WASM active' : 'WebGPU unavailable · CPU/WASM fallback';
    const loadPct = engine || cpuGenerator ? '100%' : cpuPromise ? `${cpuLoadProgress}%` : `${loadProgress}%`;
    const initMs = engine ? loadMs : cpuGenerator ? cpuLoadMs : 0;
    runtimeDetail.innerHTML = `<div><dt>Local state</dt><dd>${escapeHtml(state)}</dd></div><div><dt>Execution order</dt><dd>${escapeHtml(plannedBackend)}</dd></div><div><dt>Memory policy</dt><dd>Only one language model is loaded at a time.</dd></div><div><dt>Primary WebGPU model</dt><dd>${escapeHtml(record?.model_id || 'No compatible WebGPU model selected')}</dd></div><div><dt>Emergency CPU model</dt><dd>${escapeHtml(FAST_CPU_MODEL_ID)}</dd></div><div><dt>Selection policy</dt><dd>${escapeHtml(modelSelectionReason)}</dd></div><div><dt>WebNN detection</dt><dd>${webnnAvailable ? 'Browser API exposed · WebGPU remains preferred' : 'Not exposed by this browser'}</dd></div><div><dt>WASM threads</dt><dd>${wasmThreads}${self.crossOriginIsolated ? ' · multithread eligible' : ' · single-thread compatibility mode'}</dd></div><div><dt>Hardware utilization proof</dt><dd>${escapeHtml(measuredLocalBackend ? `${measuredLocalBackend} measured during generation on this device` : 'Waiting for the first successful local generation')}</dd></div><div><dt>Loaded model</dt><dd>${escapeHtml(engineModelId || 'Not loaded yet')}</dd></div><div><dt>Model memory target</dt><dd>${record && modelMemoryMB(record) ? `${(modelMemoryMB(record) / 1024).toFixed(2)} GB` : 'Unknown'}</dd></div><div><dt>Load progress</dt><dd>${loadPct}</dd></div><div><dt>Initialization time</dt><dd>${initMs ? `${(initMs / 1000).toFixed(2)} s` : '—'}</dd></div><div><dt>Execution thread</dt><dd>${escapeHtml(localThread)}</dd></div><div><dt>Measured local speed</dt><dd>${localTps ? `${localTps.toFixed(1)} tok/s` : '—'}</dd></div><div><dt>Measured first visible output</dt><dd>${localTtft ? `${(localTtft / 1000).toFixed(2)} s` : '—'}</dd></div><div><dt>Cache backend</dt><dd>Browser-managed model cache</dd></div><div><dt>Model storage</dt><dd>${storagePersistent ? 'Browser marked persistent' : 'Browser-managed cache'}</dd></div><div><dt>Origin storage</dt><dd>${storageQuota ? `${(storageUsage / 1024 / 1024).toFixed(0)} MB / ${(storageQuota / 1024 / 1024 / 1024).toFixed(1)} GB` : 'Unavailable'}</dd></div><div><dt>WebGPU compatibility</dt><dd>${compatible ? 'Compatible' : adapter ? 'Trying a compatible model build' : 'No adapter exposed by browser'}</dd></div><div><dt>Runtime stats</dt><dd>${escapeHtml(localRuntimeStats.slice(0, 220))}</dd></div>${runtimeError ? `<div><dt>Last local error</dt><dd>${escapeHtml(runtimeError)}</dd></div>` : ''}`;
    renderTelemetry();
};

const createEngineForRecord = async (record: ModelRecordLike, callback: (report: { progress?: number; text?: string }) => void) => {
    const runtimeAppConfig = { ...prebuiltAppConfig, cacheBackend: 'indexeddb' as const, model_list: [record] };
    try {
        localThread = 'Web Worker';
        const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
        const created = await CreateWebWorkerMLCEngine(worker, record.model_id, { appConfig: runtimeAppConfig, initProgressCallback: callback });
        engineWorker = worker;
        return created;
    } catch (workerError) {
        console.warn('TZ worker fallback', workerError);
        engineWorker?.terminate();
        engineWorker = null;
        localThread = 'Main thread fallback';
        return CreateMLCEngine(record.model_id, { appConfig: runtimeAppConfig, initProgressCallback: callback });
    }
};

const extractCpuText = (result: unknown) => {
    const first = Array.isArray(result) ? result[0] : result;
    if (!first || typeof first !== 'object') return '';
    const generated = (first as { generated_text?: unknown }).generated_text;
    if (typeof generated === 'string') return normalizeOutput(generated);
    if (Array.isArray(generated)) {
        for (let index = generated.length - 1; index >= 0; index--) {
            const message = generated[index] as { role?: unknown; content?: unknown };
            if (message?.role === 'assistant' && typeof message.content === 'string') return normalizeOutput(message.content);
        }
    }
    return '';
};

const ensureCpu = async () => {
    if (cpuGenerator) return;
    if (cpuPromise) { await cpuPromise; return; }
    cpuPromise = (async () => {
        runtimeError = '';
        cpuLoadProgress = 0;
        cpuLoadText = 'Planning local fallback';
        const started = performance.now();
        renderRuntime();
        const { pipeline, env } = await import('@huggingface/transformers');
        const nav = navigator as Navigator & { hardwareConcurrency?: number; ml?: unknown };
        webnnAvailable = Boolean(nav.ml);
        wasmThreads = self.crossOriginIsolated ? Math.max(1, Math.min(8, nav.hardwareConcurrency || 4)) : 1;
        const onnxBackend = (env.backends as any).onnx;
        if (onnxBackend?.wasm) onnxBackend.wasm.numThreads = wasmThreads;
        const routes = [{ device: 'wasm', label: 'CPU / WebAssembly q4', model: FAST_CPU_MODEL_ID, dtype: 'q4' }, { device: 'wasm', label: 'CPU / WebAssembly uint8', model: FAST_CPU_MODEL_ID, dtype: 'uint8' }];
        let lastError = 'No local fallback route succeeded';
        for (const route of routes) {
            try {
                cpuLoadProgress = 0;
                cpuLoadText = `Trying ${route.label}`;
                renderRuntime();
                const created = await pipeline('text-generation', route.model, {
                    device: route.device as any,
                    dtype: route.dtype as any,
                    progress_callback: (progress: unknown) => {
                        const info = progress as { progress?: number; status?: string; file?: string };
                        if (typeof info.progress === 'number') {
                            const normalized = info.progress <= 1 ? info.progress * 100 : info.progress;
                            cpuLoadProgress = Math.max(0, Math.min(100, Math.round(normalized)));
                        }
                        cpuLoadText = typeof info.status === 'string' ? info.status : typeof info.file === 'string' ? `Loading ${info.file}` : `Loading ${route.label} model`;
                        renderRuntime();
                    }
                });
                cpuGenerator = created as unknown as CpuGenerator;
                cpuDeviceLabel = route.label;
                cpuDtype = route.dtype;
                fastModelId = route.model;
                cpuLoadProgress = 100;
                cpuLoadMs = performance.now() - started;
                cpuLoadText = `${route.label} model resident`;
                await refreshStorage();
                break;
            } catch (error) {
                lastError = `${route.label}: ${error instanceof Error ? error.message : String(error)}`;
                cpuGenerator = null;
            }
        }
        if (!cpuGenerator) throw new Error(lastError);
    })();
    try {
        await cpuPromise;
    } catch (error) {
        cpuGenerator = null;
        runtimeError = `Fast local: ${error instanceof Error ? error.message : String(error)}`;
        cpuLoadText = 'Fast local unavailable';
        throw error;
    } finally {
        cpuPromise = null;
        renderRuntime();
    }
};

const releaseFastModel = async () => {
    const generator = cpuGenerator;
    if (!generator) return;
    try { await generator.dispose?.(); } catch { /* best-effort browser resource release */ }
    cpuGenerator = null;
    cpuLoadProgress = 0;
    cpuLoadText = 'Fast lane released for quality model';
    renderRuntime();
};

const ensureTZ = async () => {
    if (engine) return;
    if (enginePromise) { await enginePromise; return; }
    enginePromise = (async () => {
        runtimeError = '';
        loadProgress = 0;
        loadText = 'Starting local engine';
        loadStarted = performance.now();
        renderRuntime();
        if (!adapter) {
            runtimeError = 'WebGPU unavailable';
            loadText = 'Local engine unavailable';
            return;
        }
        if (!modelCandidates.length) planModels();
        let lastError = 'No compatible model';
        for (let index = 0; index < modelCandidates.length; index++) {
            const record = modelCandidates[index];
            selectedModelRecord = record;
            loadProgress = 0;
            loadText = index === 0 ? `Loading ${record.model_id}` : `Trying lighter model ${record.model_id}`;
            const callback = (report: { progress?: number; text?: string }) => {
                if (typeof report.progress === 'number') loadProgress = Math.max(0, Math.min(100, Math.round(report.progress * 100)));
                if (report.text) loadText = report.text;
                renderRuntime();
            };
            try {
                engine = await createEngineForRecord(record, callback);
                engineModelId = record.model_id;
                loadProgress = 100;
                loadMs = performance.now() - loadStarted;
                loadText = 'Local engine resident';
                runtimeError = '';
                await refreshStorage();
                break;
            } catch (error) {
                engine = null;
                engineWorker?.terminate();
                engineWorker = null;
                lastError = error instanceof Error ? error.message : String(error);
                loadText = 'Model failed; selecting a lighter local fallback';
                renderRuntime();
            }
        }
        if (!engine) {
            runtimeError = lastError;
            loadText = 'Local engine unavailable';
        }
    })();
    try {
        await enginePromise;
    } finally {
        enginePromise = null;
        renderRuntime();
    }
};

const loadHardware = async () => {
    const nav = navigator as Navigator & { deviceMemory?: number; hardwareConcurrency?: number; ml?: unknown; gpu?: { requestAdapter: (options?: Record<string, unknown>) => Promise<GPUAdapterLike | null> } };
    const entropy = await highEntropyHardware();
    webnnAvailable = Boolean(nav.ml);
    wasmThreads = self.crossOriginIsolated ? Math.max(1, Math.min(8, nav.hardwareConcurrency || 4)) : 1;
    try {
        adapter = await nav.gpu?.requestAdapter({ powerPreference: 'high-performance' }) || await nav.gpu?.requestAdapter() || null;
    } catch {
        adapter = null;
    }
    adapterFeatures = new Set(adapter?.features ? Array.from(adapter.features) : []);
    planModels();
    await refreshStorage();
    const info = adapter?.info || {};
    const limits = adapter?.limits || {};
    const exactModel = typeof entropy.model === 'string' && entropy.model ? entropy.model : 'Not exposed by this browser';
    const arch = typeof entropy.architecture === 'string' && entropy.architecture ? String(entropy.architecture) : 'Not exposed by this browser';
    const gpuIdentity = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' · ') || 'Not exposed by this browser';
    const maxBuffer = limits.maxBufferSize ? `${(limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB` : 'Withheld';
    const storageBinding = limits.maxStorageBufferBindingSize ? `${(limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB` : 'Withheld';
    const rows = [
        ['Device class', detectDeviceFamily()],
        ['Exact hardware model', exactModel],
        ['Display profile', displayProfile()],
        ['Browser', detectBrowser()],
        ['Platform', navigator.platform || 'Browser withheld'],
        ['CPU architecture', arch],
        ['Logical CPU cores', String(nav.hardwareConcurrency || 'Browser withheld')],
        ['Memory hint', nav.deviceMemory ? `${nav.deviceMemory} GB` : 'Browser withheld'],
        ['WebGPU adapter', adapter ? 'Active' : 'Unavailable'],
        ['WebNN accelerator', webnnAvailable ? 'Browser API exposed' : 'Not exposed'],
        ['CPU/WASM fallback', 'uint8 · available on first local prompt'],
        ['WASM threads', `${wasmThreads}${self.crossOriginIsolated ? ' · multithread eligible' : ' · cross-origin isolation unavailable'}`],
        ['GPU identity', gpuIdentity],
        ['GPU max buffer', maxBuffer],
        ['GPU storage binding', storageBinding],
        ['WebGPU features', adapterFeatures.size ? Array.from(adapterFeatures).sort().join(', ') : 'None exposed'],
        ['Storage persistent', storagePersistent ? 'Yes' : 'Browser managed'],
        ['Origin storage used', storageQuota ? `${(storageUsage / 1024 / 1024).toFixed(0)} MB` : 'Unavailable'],
        ['Session persistence', 'Chat + API key reset on refresh'],
        ['Online', navigator.onLine ? 'Yes' : 'No']
    ];
    hardware.innerHTML = rows.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd></div>`).join('');
    renderRuntime();
    if (!engine && !enginePromise && !cpuGenerator && !cpuPromise) setTimeout(() => { if (engine || enginePromise || cpuGenerator || cpuPromise) return; if (adapter) void ensureTZ().then(() => { if (!engine) return ensureCpu(); }).catch(() => ensureCpu().catch(() => undefined)); else void ensureCpu().catch(() => undefined); }, 300);
};

const webgpuGenerate = async (messages: Msg[], onUpdate: (text: string) => void) => {
    await ensureTZ();
    const localEngine = engine;
    if (!localEngine) throw new Error(runtimeError || 'WebGPU runtime unavailable');
    let text = '';
    let usage: { prompt_tokens?: number; completion_tokens?: number } | undefined;
    let first = 0;
    let lastPaint = 0;
    const started = performance.now();
    const recentMessages = messages.slice(-6);
    const localMessages = [{ role: 'system' as const, content: tzSystemPrompt }, ...recentMessages.map(message => ({ role: message.role, content: message.content }))];
    const lastUser = [...recentMessages].reverse().find(message => message.role === 'user')?.content || '';
    const maxTokens = /\b(code|script|function|email|letter|write|draft|explain|steps|list)\b/i.test(lastUser) ? 256 : 128;
    const stream = await localEngine.chat.completions.create({ messages: localMessages, stream: true, stream_options: { include_usage: true }, max_tokens: maxTokens, temperature: 0.35, top_p: 0.9 });
    for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content || '';
        if (delta && !first) first = performance.now();
        text += delta;
        if (chunk.usage) usage = chunk.usage;
        const now = performance.now();
        if (delta && now - lastPaint > 70) {
            onUpdate(text);
            lastPaint = now;
        }
    }
    text = normalizeOutput(text);
    if (!text) throw new Error('WebGPU model returned an empty response');
    onUpdate(text);
    const elapsed = performance.now() - started;
    const output = usage?.completion_tokens || estimateTokens(text);
    localTps = output / Math.max(0.001, elapsed / 1000);
    localTtft = first ? first - started : 0;
    measuredLocalBackend = 'WebGPU / WebLLM';
    localLane = 'Quality WebLLM';
    try {
        localRuntimeStats = await localEngine.runtimeStatsText();
    } catch {
        localRuntimeStats = 'WebLLM WebGPU runtime';
    }
    renderRuntime();
    return { text, usage, ttft: localTtft };
};

const cpuGenerate = async (messages: Msg[], onUpdate: (text: string) => void) => {
    await ensureCpu();
    const generator = cpuGenerator;
    if (!generator) throw new Error(runtimeError || 'Fast local runtime unavailable');
    const { TextStreamer } = await import('@huggingface/transformers');
    const started = performance.now();
    let first = 0;
    let streamedText = '';
    const chat: CpuChatMessage[] = [{ role: 'system', content: cpuSystemPrompt }, ...messages.slice(-3).map(message => ({ role: message.role, content: message.content }))];
    const streamer = new TextStreamer(generator.tokenizer, { skip_prompt: true, skip_special_tokens: true, callback_function: (piece: string) => {
        if (!piece) return;
        if (!first && piece.trim()) first = performance.now();
        streamedText += piece;
        const visible = normalizeOutput(streamedText);
        if (visible) onUpdate(visible);
    } });
    const lastUser = [...messages].reverse().find(message => message.role === 'user')?.content || '';
    const maxNewTokens = /\b(code|script|function|email|letter|write|draft|explain|steps|list)\b/i.test(lastUser) ? 144 : 72;
    const result = await generator(chat, { max_new_tokens: maxNewTokens, do_sample: false, repetition_penalty: 1.04, return_full_text: false, streamer });
    const elapsed = performance.now() - started;
    const resultText = extractCpuText(result);
    const text = resultText || normalizeOutput(streamedText);
    if (!text) throw new Error('Fast local model returned an empty response');
    onUpdate(text);
    const output = estimateTokens(text);
    localTps = output / Math.max(0.001, elapsed / 1000);
    localTtft = first ? first - started : elapsed;
    localThread = `Transformers.js · ${cpuDeviceLabel} · streaming`;
    localRuntimeStats = `Transformers.js ${cpuDeviceLabel} streaming · ${fastModelId} · ${cpuDtype} · ${wasmThreads} WASM thread${wasmThreads === 1 ? '' : 's'}`;
    engineModelId = fastModelId;
    measuredLocalBackend = cpuDeviceLabel;
    localLane = 'Fast Transformers.js';
    renderRuntime();
    return { text, usage: undefined, ttft: localTtft };
};

const localGenerate = async (messages: Msg[], onUpdate: (text: string) => void) => {
    let webgpuFailure = '';
    let cpuFailure = '';
    if (adapter) {
        try {
            return await webgpuGenerate(messages, onUpdate);
        } catch (error) {
            webgpuFailure = error instanceof Error ? error.message : String(error);
        }
    } else {
        webgpuFailure = 'Browser exposed no WebGPU adapter';
    }
    try {
        return await cpuGenerate(messages, onUpdate);
    } catch (error) {
        cpuFailure = error instanceof Error ? error.message : String(error);
    }
    runtimeError = [`WebGPU: ${webgpuFailure}`, `CPU/WASM: ${cpuFailure}`].join(' · ');
    throw new Error(runtimeError);
};

const externalGenerate = async (messages: Msg[]) => {
    if (!apiConfig) throw new Error('No external fallback configured');
    const endpoint = `${apiConfig.endpoint.replace(/\/$/, '')}/chat/completions`;
    const started = performance.now();
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiConfig.key}` },
        body: JSON.stringify({ model: apiConfig.model, messages: [{ role: 'system', content: tzSystemPrompt }, ...messages.map(message => ({ role: message.role, content: message.content }))], temperature: 0.35 })
    });
    if (!response.ok) throw new Error(`Provider returned ${response.status}`);
    const data = await response.json() as { choices?: Array<{ message?: { content?: string } }>; usage?: { prompt_tokens?: number; completion_tokens?: number } };
    const text = normalizeOutput(data.choices?.[0]?.message?.content || '');
    if (!text) throw new Error('External provider returned an empty response');
    return { text, usage: data.usage, ttft: 0, elapsed: performance.now() - started };
};

const addInstant = (text: string, result: string) => {
    errorEl.textContent = '';
    history.push({ role: 'user', content: text });
    prompt.value = '';
    const id = turns.length ? Math.max(...turns.map(turn => turn.id)) + 1 : 1;
    history.push({ role: 'assistant', content: result, turn: id });
    lastRoute = 'Instant';
    turns.push({ id, route: 'Instant', model: 'TZ instant tool', input: estimateTokens(text), output: estimateTokens(result), ms: 0, tps: 0, ttft: 0 });
    renderMessages();
    renderTelemetry();
    renderRuntime();
    prompt.focus();
};

form.addEventListener('submit', async event => {
    event.preventDefault();
    const text = prompt.value.trim();
    errorEl.textContent = '';
    if (!text) return;
    const instant = instantTool(text);
    if (instant !== null) {
        addInstant(text, instant);
        return;
    }
    history.push({ role: 'user', content: text });
    prompt.value = '';
    const id = turns.length ? Math.max(...turns.map(turn => turn.id)) + 1 : 1;
    const assistant: Msg = { role: 'assistant', content: '', turn: id, pending: true };
    history.push(assistant);
    renderMessages();
    prompt.disabled = true;
    sendBtn.disabled = true;
    const start = performance.now();
    const context = history.slice(0, -1);
    let usage: { prompt_tokens?: number; completion_tokens?: number } | undefined;
    let ttft = 0;
    let elapsedOverride = 0;
    try {
        try {
            const local = await localGenerate(context, partial => {
                assistant.content = partial;
                assistant.pending = false;
                renderMessages();
            });
            assistant.content = local.text;
            assistant.pending = false;
            usage = local.usage;
            ttft = local.ttft;
            lastRoute = 'On-device';
        } catch (localError) {
            runtimeError = localError instanceof Error ? localError.message : String(localError);
            if (!apiConfig) throw localError;
            const external = await externalGenerate(context);
            assistant.content = external.text;
            assistant.pending = false;
            usage = external.usage;
            elapsedOverride = external.elapsed;
            lastRoute = 'External';
        }
        const ms = elapsedOverride || performance.now() - start;
        const input = usage?.prompt_tokens || estimateTokens(context.map(message => message.content).join('\n'));
        const output = usage?.completion_tokens || estimateTokens(assistant.content);
        const tps = lastRoute === 'On-device' ? localTps : lastRoute === 'External' ? output / Math.max(0.001, ms / 1000) : 0;
        turns.push({ id, route: lastRoute, model: lastRoute === 'On-device' ? engineModelId : lastRoute === 'External' ? apiConfig?.model || 'External' : 'TZ instant tool', input, output, ms, tps, ttft });
        renderMessages();
        renderTelemetry();
        renderRuntime();
    } catch (error) {
        history.pop();
        runtimeError = error instanceof Error ? error.message : String(error);
        errorEl.textContent = apiConfig ? 'Both the local engine and your external fallback failed. Open Monitoring to retry or copy diagnostics.' : 'The local engine could not complete this request. Open Monitoring, tap Retry local engine, or copy diagnostics for support.';
        renderMessages();
        renderRuntime();
    } finally {
        prompt.disabled = false;
        sendBtn.disabled = false;
        prompt.focus();
    }
});

const clearSession = () => {
    history = [];
    turns = [];
    lastRoute = 'Instant';
    errorEl.textContent = '';
    renderMessages();
    renderTelemetry();
    renderRuntime();
};
$<HTMLButtonElement>('clearBtn').addEventListener('click', clearSession);

const runtimeDiagnostics = () => {
    const nav = navigator as Navigator & { deviceMemory?: number; hardwareConcurrency?: number; gpu?: unknown };
    return [
        'TZ local runtime diagnostic',
        `Captured: ${new Date().toISOString()}`,
        `Secure context: ${isSecureContext}`,
        `Device class: ${detectDeviceFamily()}`,
        `Browser: ${navigator.userAgent}`,
        `WebGPU API exposed: ${Boolean(nav.gpu)}`,
        `WebGPU adapter acquired: ${Boolean(adapter)}`,
        `Adapter features: ${adapterFeatures.size ? Array.from(adapterFeatures).sort().join(', ') : 'none exposed'}`,
        `Selected model: ${selectedModelRecord?.model_id || 'none'}`,
        `Loaded model: ${engineModelId || 'none'}`,
        `Execution backend: ${measuredLocalBackend || 'not measured'}`,
        `Cross-origin isolated: ${self.crossOriginIsolated}`,
        `Logical cores: ${nav.hardwareConcurrency || 'withheld'}`,
        `Memory hint: ${nav.deviceMemory ? `${nav.deviceMemory} GB` : 'withheld'}`,
        `Online: ${navigator.onLine}`,
        `Last local error: ${runtimeError || 'none'}`
    ].join('\n');
};

const retryLocalEngine = async () => {
    const retryButton = $<HTMLButtonElement>('retryRuntimeBtn');
    const diagnosticStatus = $<HTMLElement>('diagnosticStatus');
    if (enginePromise || cpuPromise) { diagnosticStatus.textContent = 'TZ is still loading. Current progress is shown above.'; return; }
    retryButton.disabled = true;
    diagnosticStatus.textContent = 'Restarting the local engine…';
    try {
        const previousEngine = engine as (MLCEngineInterface & { unload?: () => Promise<void> }) | null;
        engine = null;
        engineModelId = '';
        engineWorker?.terminate();
        engineWorker = null;
        try { await previousEngine?.unload?.(); } catch { /* best-effort GPU cleanup */ }
        const previousCpu = cpuGenerator;
        cpuGenerator = null;
        try { await previousCpu?.dispose?.(); } catch { /* best-effort WASM cleanup */ }
        runtimeError = '';
        loadProgress = 0;
        cpuLoadProgress = 0;
        await loadHardware();
        if (adapter) await ensureTZ();
        if (!engine) await ensureCpu();
        if (!engine && !cpuGenerator) throw new Error(runtimeError || 'No local execution path became ready');
        diagnosticStatus.textContent = engine ? 'Local WebGPU engine ready.' : 'Local CPU/WASM fallback ready.';
    } catch (error) {
        runtimeError = error instanceof Error ? error.message : String(error);
        diagnosticStatus.textContent = `Retry failed: ${runtimeError}`;
    } finally {
        retryButton.disabled = false;
        renderRuntime();
    }
};
$<HTMLButtonElement>('retryRuntimeBtn').addEventListener('click', () => void retryLocalEngine());
$<HTMLButtonElement>('copyDiagnosticsBtn').addEventListener('click', async () => {
    await copyText(runtimeDiagnostics());
    const diagnosticStatus = $<HTMLElement>('diagnosticStatus');
    diagnosticStatus.textContent = 'Diagnostics copied. No prompt text or API key is included.';
});

const syncApiPanel = () => {
    connectBtn.setAttribute('aria-expanded', String(!connectPanel.hidden));
    disconnectApiBtn.disabled = !apiConfig;
    apiStatus.textContent = apiConfig ? `External fallback connected: ${apiConfig.model}. TZ still tries instant and on-device execution first; prompts leave this device only if local execution fails.` : 'Optional fallback. Key stays in this browser tab and is discarded on refresh.';
    $<HTMLElement>('apiKeyAudit').textContent = apiConfig ? 'Fallback key in memory · refresh clears it' : 'None';
    renderRuntime();
};
connectBtn.addEventListener('click', () => {
    connectPanel.hidden = !connectPanel.hidden;
    syncApiPanel();
    if (!connectPanel.hidden) apiEndpoint.focus();
});
apiEndpoint.addEventListener('input', () => {
    const value = apiEndpoint.value.trim();
    if (/^http:\/\//i.test(value)) {
        apiStatus.textContent = 'External AI endpoints must use HTTPS.';
        apiEndpoint.setCustomValidity('External AI endpoints must use HTTPS.');
    } else {
        apiEndpoint.setCustomValidity('');
        if (!apiConfig) apiStatus.textContent = 'Optional fallback. Key stays in this browser tab only.';
    }
});
apiForm.addEventListener('submit', event => {
    event.preventDefault();
    const endpoint = apiEndpoint.value.trim().replace(/\/$/, '');
    const model = apiModel.value.trim();
    const key = apiKey.value.trim();
    if (!endpoint || !model || !key) {
        apiStatus.textContent = 'Endpoint, model, and API key are required.';
        return;
    }
    let parsed: URL;
    try {
        parsed = new URL(endpoint);
    } catch {
        apiStatus.textContent = 'Enter a valid HTTPS endpoint URL.';
        return;
    }
    if (parsed.protocol !== 'https:') {
        apiStatus.textContent = 'External AI endpoints must use HTTPS.';
        return;
    }
    apiConfig = { endpoint, model, key };
    apiKey.value = '';
    apiStatus.textContent = 'External fallback connected. TZ remains local-first.';
    syncApiPanel();
});
disconnectApiBtn.addEventListener('click', () => {
    apiConfig = null;
    apiStatus.textContent = 'External fallback disconnected.';
    syncApiPanel();
});

const navButtons = [...document.querySelectorAll<HTMLButtonElement>('.nav')];
navButtons.forEach(button => button.addEventListener('click', () => {
    navButtons.forEach(item => item.classList.toggle('active', item === button));
    document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
    document.getElementById(`${button.dataset.view}View`)?.classList.add('active');
}));

syncApiPanel();
renderMessages();
renderTelemetry();
renderRuntime();
void loadHardware();
addEventListener('online', () => void loadHardware());
addEventListener('pageshow', event => {
    if ((event as PageTransitionEvent).persisted) {
        apiConfig = null;
        clearSession();
        syncApiPanel();
    }
    void loadHardware();
});
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && !engine && !enginePromise) void loadHardware();
});