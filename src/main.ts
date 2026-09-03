import './styles.css';
import type { MLCEngineInterface } from '@mlc-ai/web-llm';
import type { PreTrainedTokenizer } from '@huggingface/transformers';

type Role = 'user' | 'assistant';
type ChatRole = Role | 'system';
type Msg = { role: Role; content: string; turn?: number; pending?: boolean };
type Route = 'Instant' | 'On-device' | 'Connected AI';
type RoutingMode = 'auto' | 'local' | 'smart';
type Turn = { id: number; route: Route; model: string; input: number; output: number; ms: number; tps: number; ttft: number };
type ApiConfig = { provider: string; endpoint: string; model: string; key: string };
type ProviderPreset = { name: string; endpoint: string; model: string; note: string };
type GPUAdapterLike = { info?: { vendor?: string; architecture?: string; device?: string; description?: string }; features?: Set<string>; limits?: Record<string, number> };
type WebLLMModule = typeof import('@mlc-ai/web-llm');
type ModelRecordLike = WebLLMModule['prebuiltAppConfig']['model_list'][number];
type ChatMessage = { role: ChatRole; content: string };
type CpuGenerator = ((input: ChatMessage[], options?: Record<string, unknown>) => Promise<unknown>) & { tokenizer: PreTrainedTokenizer; dispose?: () => Promise<void> };
type GenerationResult = { text: string; usage?: { prompt_tokens?: number; completion_tokens?: number }; ttft: number; elapsed: number };

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
const apiPreset = $<HTMLSelectElement>('apiPreset');
const apiEndpoint = $<HTMLInputElement>('apiEndpoint');
const apiModel = $<HTMLInputElement>('apiModel');
const apiKey = $<HTMLInputElement>('apiKey');
const apiStatus = $<HTMLElement>('apiStatus');
const routeMode = $<HTMLSelectElement>('routeMode');
const disconnectApiBtn = $<HTMLButtonElement>('disconnectApiBtn');

const PROVIDERS: Record<string, ProviderPreset> = {
    openrouter: {
        name: 'OpenRouter free',
        endpoint: 'https://openrouter.ai/api/v1',
        model: 'openrouter/free',
        note: 'Routes to an available free model. A free OpenRouter account and API key are required; free limits and availability can change.',
    },
    groq: {
        name: 'Groq free tier',
        endpoint: 'https://api.groq.com/openai/v1',
        model: 'openai/gpt-oss-20b',
        note: 'Fast free developer quota with a Groq account and API key. Your Groq organization limits apply.',
    },
    gemini: {
        name: 'Gemini free tier',
        endpoint: 'https://generativelanguage.googleapis.com/v1beta/openai',
        model: 'gemini-2.5-flash-lite',
        note: 'Uses Google AI Studio free-tier quota with your own API key. Google account and free-tier terms apply.',
    },
    custom: {
        name: 'Custom compatible API',
        endpoint: '',
        model: '',
        note: 'Use an HTTPS service with an OpenAI-compatible chat completions endpoint.',
    },
};

const escapeHtml = (value: string) => value.replace(/[&<>\"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', "'": '&#039;' }[char] || char));
const estimateTokens = (value: string) => Math.max(1, Math.ceil(value.length / 4));
const normalizeOutput = (value: string) => value
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .trim()
    .replace(/^(?:assistant|tz)\s*:\s*/i, '')
    .replace(/^```[\w-]*\s*/, '')
    .replace(/\s*```$/, '')
    .trim();

const tzSystemPrompt = [
    'You are TZ, a precise, practical general-purpose assistant running for the person using this device.',
    'Use the conversation context, including earlier details, before answering. Never pretend context was not provided.',
    'Lead with the useful answer. Be concise by default, but fully complete writing, coding, planning, and explanation requests.',
    'For code, return complete runnable code rather than pseudocode. For writing, return polished text ready to use.',
    'Do not invent current facts. If live knowledge is required and unavailable, say that plainly.',
    'Avoid generic filler, repeated disclaimers, fake enthusiasm, and phrases such as “I would be happy to assist.”',
    'Do not mention model providers or runtime details unless the user asks.',
].join(' ');

const cpuSystemPrompt = `${tzSystemPrompt} Keep the response compact because this is the emergency low-resource local path.`;
const MOBILE_GPU_MODELS = [
    'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
    'Qwen2.5-1.5B-Instruct-q4f32_1-MLC',
    'Qwen3-1.7B-q4f16_1-MLC',
    'Qwen3-1.7B-q4f32_1-MLC',
    'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    'Qwen2.5-0.5B-Instruct-q4f32_1-MLC',
];
const DESKTOP_GPU_MODELS = [
    'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    'Qwen2.5-3B-Instruct-q4f32_1-MLC',
    ...MOBILE_GPU_MODELS,
];
const FAST_CPU_MODEL_ID = 'onnx-community/SmolLM2-360M-Instruct-ONNX';

let engine: MLCEngineInterface | null = null;
let webllmModule: WebLLMModule | null = null;
let webllmPromise: Promise<WebLLMModule> | null = null;
let enginePromise: Promise<void> | null = null;
let engineWorker: Worker | null = null;
let history: Msg[] = [];
let turns: Turn[] = [];
let runtimeError = '';
let apiConfig: ApiConfig | null = null;
let routingMode: RoutingMode = 'auto';
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
let cpuGenerator: CpuGenerator | null = null;
let cpuPromise: Promise<void> | null = null;
let cpuLoadProgress = 0;
let cpuLoadText = 'Low-resource local idle';
let cpuLoadMs = 0;
let measuredLocalBackend = '';
let cpuDeviceLabel = 'CPU / WebAssembly';
let cpuDtype = 'q4';
let webnnAvailable = false;
let wasmThreads = 1;
let lastContextMessages = 0;
let lastContextTokens = 0;
let isGenerating = false;
let cancelRequested = false;
let externalAbort: AbortController | null = null;
let generationId = 0;
let runtimeRenderQueued = false;

const isMobileDevice = () => /iPhone|iPad|Android/i.test(navigator.userAgent)
    || (navigator.maxTouchPoints > 1 && Math.min(screen.width, screen.height) < 900);

const calculateLocal = (input: string): string | null => {
    const normalized = input.trim().replace(/^(?:what is|what's|calculate|compute|solve)\s+/i, '').replace(/[×x]/gi, '*').replace(/÷/g, '/').replace(/\?$/, '').trim();
    if (!normalized || !/^[0-9+\-*/().%\s]+$/.test(normalized)) return null;
    const tokens = normalized.match(/\d+(?:\.\d+)?|[()+\-*/%]/g);
    if (!tokens) return null;
    let index = 0;
    const expression = (): number => {
        let value = term();
        while (tokens[index] === '+' || tokens[index] === '-') {
            const operator = tokens[index++];
            const right = term();
            value = operator === '+' ? value + right : value - right;
        }
        return value;
    };
    const term = (): number => {
        let value = factor();
        while (tokens[index] === '*' || tokens[index] === '/' || tokens[index] === '%') {
            const operator = tokens[index++];
            const right = factor();
            if ((operator === '/' || operator === '%') && right === 0) throw new Error('division by zero');
            value = operator === '*' ? value * right : operator === '/' ? value / right : value % right;
        }
        return value;
    };
    const factor = (): number => {
        if (tokens[index] === '+') { index++; return factor(); }
        if (tokens[index] === '-') { index++; return -factor(); }
        if (tokens[index] === '(') {
            index++;
            const value = expression();
            if (tokens[index++] !== ')') throw new Error('unclosed parenthesis');
            return value;
        }
        const value = Number(tokens[index++]);
        if (!Number.isFinite(value)) throw new Error('invalid number');
        return value;
    };
    try {
        const result = expression();
        if (index !== tokens.length || !Number.isFinite(result)) return null;
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
    const calculation = calculateLocal(input);
    if (calculation !== null) return calculation;
    const text = input.trim();
    if (/^test[.!?]*$/i.test(text)) return 'TZ is online and ready.';
    if (/^(?:what(?:'s| is) )?(?:the )?(?:date|today's date)(?: today)?[?!.]*$/i.test(text)) {
        return new Intl.DateTimeFormat(undefined, { dateStyle: 'full' }).format(new Date());
    }
    if (/^(?:what time is it|what(?:'s| is) the time|current time)[?!.]*$/i.test(text)) {
        return new Intl.DateTimeFormat(undefined, { timeStyle: 'short', timeZoneName: 'short' }).format(new Date());
    }
    const wordCount = text.match(/^word count\s*:\s*([\s\S]+)/i);
    if (wordCount) return `${wordCount[1].trim().split(/\s+/).filter(Boolean).length} words`;
    const style = text.match(/^write\s+(?:the\s+name\s+)?(.+?)\s+and\s+make\s+it\s+look\s+(?:cool|stylish|fancy)[.!?]*$/i);
    if (style) {
        const value = style[1].replace(/^['\"]|['\"]$/g, '').trim();
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

const nearMessageBottom = () => messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 160;

const buildMessageElement = (message: Msg) => {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${message.role}`;
    if (message.turn) wrapper.dataset.turn = String(message.turn);
    if (message.role === 'user') {
        wrapper.textContent = message.content;
        return wrapper;
    }
    if (message.pending && !message.content) {
        const thinking = document.createElement('div');
        thinking.className = 'thinking';
        thinking.textContent = engine || apiConfig ? 'Thinking…' : 'Preparing the local model…';
        wrapper.appendChild(thinking);
        return wrapper;
    }
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
        setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
    });
    toolbar.append(label, copy);
    const pre = document.createElement('pre');
    pre.className = 'answer-code';
    const code = document.createElement('code');
    code.textContent = message.content;
    pre.appendChild(code);
    card.append(toolbar, pre);
    wrapper.appendChild(card);
    if (message.turn && !message.pending) {
        const turn = turns.find(item => item.id === message.turn);
        if (turn) {
            const meta = document.createElement('div');
            meta.className = 'turn-meta';
            const speed = turn.tps > 0 ? ` · ${turn.tps.toFixed(1)} tok/s` : '';
            meta.textContent = `${turn.output} output tokens${speed} · ${turn.route}`;
            wrapper.appendChild(meta);
        }
    }
    return wrapper;
};

const renderMessages = () => {
    messagesEl.replaceChildren();
    document.body.classList.toggle('has-chat', history.length > 0);
    if (!history.length) {
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = 'Ask a question, write something, plan, calculate, or code.';
        messagesEl.appendChild(empty);
        return;
    }
    const fragment = document.createDocumentFragment();
    history.forEach(message => fragment.appendChild(buildMessageElement(message)));
    messagesEl.appendChild(fragment);
    messagesEl.scrollTop = messagesEl.scrollHeight;
};

const updateAssistant = (message: Msg) => {
    const shouldScroll = nearMessageBottom();
    const wrapper = message.turn ? messagesEl.querySelector<HTMLElement>(`.message[data-turn='${message.turn}']`) : null;
    const code = wrapper?.querySelector<HTMLElement>('.answer-code code');
    if (!wrapper || !code) {
        renderMessages();
        return;
    }
    code.textContent = message.content;
    if (shouldScroll) messagesEl.scrollTop = messagesEl.scrollHeight;
};

const modeLabel = () => routingMode === 'local' ? 'Local only' : routingMode === 'smart' ? 'Connected AI first' : 'Auto routing';

const renderTelemetry = () => {
    const total = turns.reduce((sum, turn) => sum + turn.input + turn.output, 0);
    const last = turns.at(-1);
    $<HTMLElement>('mTokens').textContent = `${total.toLocaleString()} session tokens`;
    $<HTMLElement>('mLocalSpeed').textContent = localTps > 0 ? `${localTps.toFixed(1)}` : '—';
    $<HTMLElement>('mLatency').textContent = last ? `${(last.ms / 1000).toFixed(2)}s` : '—';
    $<HTMLElement>('mFirstToken').textContent = last && last.ttft > 0 ? `first token ${(last.ttft / 1000).toFixed(2)}s` : 'first token —';
    $<HTMLElement>('mGpu').textContent = engine ? 'WEBGPU' : cpuGenerator ? 'WASM' : adapter ? 'WEBGPU' : 'OFF';
    $<HTMLElement>('mLoad').textContent = engine || cpuGenerator ? '100%' : enginePromise ? `${loadProgress}%` : cpuPromise ? `${cpuLoadProgress}%` : runtimeError ? 'RETRY' : '0%';
    $<HTMLElement>('mLoadText').textContent = engine ? `loaded in ${(loadMs / 1000).toFixed(1)}s` : cpuGenerator ? `loaded in ${(cpuLoadMs / 1000).toFixed(1)}s` : cpuPromise ? cpuLoadText.slice(0, 48) : loadText.slice(0, 48);
    $<HTMLElement>('mContext').textContent = lastContextMessages ? `${lastContextMessages}` : '—';
    $<HTMLElement>('mContextText').textContent = lastContextTokens ? `messages · ~${lastContextTokens} tokens` : 'messages used last turn';
    $<HTMLElement>('chatMode').textContent = modeLabel();
    ledger.innerHTML = turns.length
        ? turns.map(turn => `<tr><td>${turn.id}</td><td>${escapeHtml(turn.model)}</td><td>${turn.input}</td><td>${turn.output}</td><td>${turn.ms ? `${(turn.ms / 1000).toFixed(2)}s` : 'instant'}</td><td>${turn.tps > 0 ? turn.tps.toFixed(1) : '—'}</td><td>${escapeHtml(turn.route)}</td></tr>`).join('')
        : '<tr><td colspan="7">No turns yet.</td></tr>';
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
    if (/iPad/i.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)) return 'Apple iPad';
    if (/Android/i.test(ua)) return 'Android device';
    if (/Mac/i.test(navigator.platform)) return 'Apple Mac';
    if (/Win/i.test(navigator.platform)) return 'Windows PC';
    if (/Linux/i.test(navigator.platform)) return 'Linux device';
    return 'Current device';
};

const displayProfile = () => `${Math.min(screen.width, screen.height)}×${Math.max(screen.width, screen.height)} CSS px @ ${devicePixelRatio.toFixed(1)}x`;

const highEntropyHardware = async () => {
    const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { getHighEntropyValues?: (keys: string[]) => Promise<Record<string, unknown>> } };
    try {
        return await nav.userAgentData?.getHighEntropyValues?.(['architecture', 'bitness', 'model']) || {};
    } catch {
        return {};
    }
};

const modelMemoryMB = (record: ModelRecordLike) => Number(record.vram_required_MB || 0);

const loadWebLLM = async () => {
    if (webllmModule) return webllmModule;
    if (!webllmPromise) webllmPromise = import('@mlc-ai/web-llm');
    webllmModule = await webllmPromise;
    return webllmModule;
};

const modelIsCompatible = (record: ModelRecordLike, limits: Record<string, number>) => {
    const required = Array.isArray(record.required_features) ? record.required_features.map(String) : [];
    if (!required.every(feature => adapterFeatures.has(feature))) return false;
    const storageLimit = Number(limits.maxStorageBufferBindingSize || 0);
    const requiredBuffer = Number(record.buffer_size_required_bytes || 0);
    if (storageLimit && requiredBuffer && requiredBuffer > storageLimit) return false;
    return !/vision/i.test(record.model_id);
};

const planModels = () => {
    if (!webllmModule) {
        selectedModelRecord = null;
        modelCandidates = [];
        modelBudgetMB = 0;
        modelSelectionReason = 'Runtime catalog loads after the interface becomes interactive';
        return;
    }
    const limits = adapter?.limits || {};
    const compatible = webllmModule.prebuiltAppConfig.model_list.filter(record => modelIsCompatible(record, limits));
    const preferredIds = isMobileDevice() ? MOBILE_GPU_MODELS : DESKTOP_GPU_MODELS;
    const preferred = preferredIds.map(modelId => compatible.find(record => record.model_id === modelId)).filter((record): record is ModelRecordLike => Boolean(record));
    const knownFallbacks = compatible
        .filter(record => /Qwen(?:2\.5|3)-(?:3B|1\.7B|1\.5B|0\.6B|0\.5B).*Instruct|Qwen3-1\.7B/i.test(record.model_id))
        .sort((a, b) => modelMemoryMB(a) - modelMemoryMB(b));
    modelCandidates = [...preferred, ...knownFallbacks]
        .filter((record, index, all) => all.findIndex(item => item.model_id === record.model_id) === index)
        .slice(0, 5);
    selectedModelRecord = modelCandidates[0] || null;
    modelBudgetMB = selectedModelRecord ? Math.ceil(modelMemoryMB(selectedModelRecord)) : 0;
    modelSelectionReason = selectedModelRecord
        ? `${isMobileDevice() ? 'mobile quality profile' : 'desktop quality profile'} · compatible WebGPU build · one model resident`
        : 'No supported WebLLM model was found in this runtime build';
};

const refreshStorage = async (requestPersistence = false) => {
    try {
        if (requestPersistence) await navigator.storage.persist?.();
        storagePersistent = await navigator.storage.persisted();
        const estimate = await navigator.storage.estimate();
        storageUsage = estimate.usage || 0;
        storageQuota = estimate.quota || 0;
    } catch {
        storagePersistent = false;
    }
};

const renderRuntimeNow = () => {
    const record = selectedModelRecord;
    const required = record && Array.isArray(record.required_features) ? record.required_features.map(String) : [];
    const compatible = Boolean(adapter) && required.every(feature => adapterFeatures.has(feature));
    const localReady = Boolean(engine || cpuGenerator);
    const cloudReady = Boolean(apiConfig) && routingMode !== 'local';
    const state = engine ? 'Quality WebGPU model resident and ready'
        : cpuGenerator ? 'Low-resource CPU/WASM model resident and ready'
            : enginePromise ? 'Loading the quality WebGPU model'
                : cpuPromise ? 'Loading the low-resource CPU/WASM model'
                    : runtimeError ? 'Local runtime needs a retry'
                        : adapter ? 'WebGPU model queued for idle warmup' : 'WebGPU unavailable · CPU/WASM queued';
    status.textContent = localReady ? 'TZ ready · local intelligence active'
        : cloudReady ? 'TZ ready · connected intelligence available while local warms'
            : enginePromise ? `Warming local intelligence · ${loadProgress}%`
                : cpuPromise ? `Preparing low-resource local · ${cpuLoadProgress}%`
                    : runtimeError ? 'Local runtime needs a retry' : 'TZ ready · local model warming shortly';
    $<HTMLElement>('activeModelAudit').textContent = engine ? engineModelId : cpuGenerator ? FAST_CPU_MODEL_ID : record?.model_id || FAST_CPU_MODEL_ID;
    $<HTMLElement>('localStatus').textContent = engine ? 'Local · WebGPU ready'
        : cpuGenerator ? 'Local · CPU/WASM ready'
            : enginePromise ? `WebGPU loading ${loadProgress}%`
                : cpuPromise ? `CPU/WASM loading ${cpuLoadProgress}%`
                    : runtimeError ? cloudReady ? 'Connected AI ready · local retry available' : 'Local runtime retry needed'
                        : adapter ? 'WebGPU detected · warmup queued' : 'WebGPU unavailable · CPU fallback queued';
    $<HTMLElement>('gpuAudit').textContent = engine ? 'WebGPU active on this device' : adapter ? 'WebGPU detected' : cpuGenerator ? 'CPU/WASM active' : 'WebGPU unavailable';
    const loadPercent = engine || cpuGenerator ? '100%' : cpuPromise ? `${cpuLoadProgress}%` : `${loadProgress}%`;
    const initializationMs = engine ? loadMs : cpuGenerator ? cpuLoadMs : 0;
    const connected = apiConfig ? `${apiConfig.provider} · ${apiConfig.model} · ${modeLabel()}` : 'None · local only until connected';
    runtimeDetail.innerHTML = `<div><dt>Local state</dt><dd>${escapeHtml(state)}</dd></div><div><dt>Primary model</dt><dd>${escapeHtml(record?.model_id || 'No compatible WebGPU model selected')}</dd></div><div><dt>Emergency model</dt><dd>${escapeHtml(FAST_CPU_MODEL_ID)}</dd></div><div><dt>Selection policy</dt><dd>${escapeHtml(modelSelectionReason)}</dd></div><div><dt>Context policy</dt><dd>Token-budgeted recent conversation plus compact earlier-turn memory</dd></div><div><dt>Connected intelligence</dt><dd>${escapeHtml(connected)}</dd></div><div><dt>Memory policy</dt><dd>Only one local language model is resident at a time</dd></div><div><dt>Hardware utilization proof</dt><dd>${escapeHtml(measuredLocalBackend ? `${measuredLocalBackend} measured during generation` : 'Waiting for the first successful local generation')}</dd></div><div><dt>Loaded model</dt><dd>${escapeHtml(engineModelId || 'Not loaded yet')}</dd></div><div><dt>Model memory target</dt><dd>${modelBudgetMB ? `${(modelBudgetMB / 1024).toFixed(2)} GB` : 'Unknown'}</dd></div><div><dt>Load progress</dt><dd>${loadPercent}</dd></div><div><dt>Initialization time</dt><dd>${initializationMs ? `${(initializationMs / 1000).toFixed(2)} s` : '—'}</dd></div><div><dt>Execution thread</dt><dd>${escapeHtml(localThread)}</dd></div><div><dt>Measured local speed</dt><dd>${localTps ? `${localTps.toFixed(1)} tok/s` : '—'}</dd></div><div><dt>Measured first output</dt><dd>${localTtft ? `${(localTtft / 1000).toFixed(2)} s` : '—'}</dd></div><div><dt>Model storage</dt><dd>${storagePersistent ? 'Persistent browser cache granted' : 'Browser-managed cache'}</dd></div><div><dt>Origin storage</dt><dd>${storageQuota ? `${(storageUsage / 1024 / 1024).toFixed(0)} MB / ${(storageQuota / 1024 / 1024 / 1024).toFixed(1)} GB` : 'Unavailable'}</dd></div><div><dt>WebGPU compatibility</dt><dd>${compatible ? 'Compatible' : adapter ? 'Trying compatible fallback builds' : 'No adapter exposed by browser'}</dd></div><div><dt>Runtime stats</dt><dd>${escapeHtml(localRuntimeStats.slice(0, 220))}</dd></div>${runtimeError ? `<div><dt>Last local error</dt><dd>${escapeHtml(runtimeError)}</dd></div>` : ''}`;
    renderTelemetry();
};

const renderRuntime = () => {
    if (runtimeRenderQueued) return;
    runtimeRenderQueued = true;
    requestAnimationFrame(() => {
        runtimeRenderQueued = false;
        renderRuntimeNow();
    });
};

const createEngineForRecord = async (record: ModelRecordLike, callback: (report: { progress?: number; text?: string }) => void) => {
    const mlc = await loadWebLLM();
    const runtimeAppConfig = { ...mlc.prebuiltAppConfig, cacheBackend: 'indexeddb' as const, model_list: [record] };
    localThread = 'Dedicated Web Worker';
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    try {
        const created = await mlc.CreateWebWorkerMLCEngine(worker, record.model_id, { appConfig: runtimeAppConfig, initProgressCallback: callback });
        engineWorker = worker;
        return created;
    } catch (error) {
        worker.terminate();
        throw error;
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
        cpuLoadText = 'Planning low-resource local path';
        const started = performance.now();
        renderRuntime();
        const { pipeline, env } = await import('@huggingface/transformers');
        const nav = navigator as Navigator & { hardwareConcurrency?: number; ml?: unknown };
        webnnAvailable = Boolean(nav.ml);
        wasmThreads = self.crossOriginIsolated ? Math.max(1, Math.min(8, nav.hardwareConcurrency || 4)) : 1;
        const onnxBackend = (env.backends as { onnx?: { wasm?: { numThreads?: number; proxy?: boolean } } }).onnx;
        if (onnxBackend?.wasm) {
            onnxBackend.wasm.numThreads = wasmThreads;
            onnxBackend.wasm.proxy = true;
        }
        const routes: Array<{ label: string; dtype: 'q4' | 'uint8' }> = [
            { label: 'CPU / WebAssembly q4', dtype: 'q4' },
            { label: 'CPU / WebAssembly uint8', dtype: 'uint8' },
        ];
        let lastError = 'No low-resource local route succeeded';
        for (const route of routes) {
            try {
                cpuLoadProgress = 0;
                cpuLoadText = `Trying ${route.label}`;
                const created = await pipeline('text-generation', FAST_CPU_MODEL_ID, {
                    device: 'wasm',
                    dtype: route.dtype,
                    progress_callback: (progress: unknown) => {
                        const info = progress as { progress?: number; status?: string; file?: string };
                        if (typeof info.progress === 'number') {
                            const normalized = info.progress <= 1 ? info.progress * 100 : info.progress;
                            cpuLoadProgress = Math.max(0, Math.min(100, Math.round(normalized)));
                        }
                        cpuLoadText = typeof info.status === 'string' ? info.status : typeof info.file === 'string' ? `Loading ${info.file}` : `Loading ${route.label}`;
                        renderRuntime();
                    },
                });
                cpuGenerator = created as unknown as CpuGenerator;
                cpuDeviceLabel = route.label;
                cpuDtype = route.dtype;
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
        runtimeError = `Low-resource local: ${error instanceof Error ? error.message : String(error)}`;
        cpuLoadText = 'Low-resource local unavailable';
        throw error;
    } finally {
        cpuPromise = null;
        renderRuntime();
    }
};

const ensureTZ = async () => {
    if (engine) return;
    if (enginePromise) { await enginePromise; return; }
    enginePromise = (async () => {
        runtimeError = '';
        loadProgress = 0;
        loadText = 'Starting local engine';
        loadStarted = performance.now();
        if (!adapter) throw new Error('WebGPU unavailable');
        await loadWebLLM();
        if (!modelCandidates.length) planModels();
        let lastError = 'No compatible model';
        for (let index = 0; index < modelCandidates.length; index++) {
            const record = modelCandidates[index];
            selectedModelRecord = record;
            modelBudgetMB = Math.ceil(modelMemoryMB(record));
            loadProgress = 0;
            loadText = index === 0 ? `Loading ${record.model_id}` : `Trying fallback ${record.model_id}`;
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
                loadText = 'Quality local engine resident';
                runtimeError = '';
                await refreshStorage();
                break;
            } catch (error) {
                engine = null;
                engineWorker?.terminate();
                engineWorker = null;
                lastError = error instanceof Error ? error.message : String(error);
                renderRuntime();
            }
        }
        if (!engine) throw new Error(lastError);
    })();
    try {
        await enginePromise;
    } catch (error) {
        runtimeError = error instanceof Error ? error.message : String(error);
        loadText = 'Quality local engine unavailable';
        throw error;
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
        const options = isMobileDevice() ? undefined : { powerPreference: 'high-performance' };
        adapter = await nav.gpu?.requestAdapter(options) || null;
    } catch {
        adapter = null;
    }
    adapterFeatures = new Set(adapter?.features ? Array.from(adapter.features) : []);
    planModels();
    await refreshStorage();
    const info = adapter?.info || {};
    const limits = adapter?.limits || {};
    const exactModel = typeof entropy.model === 'string' && entropy.model ? entropy.model : 'Not exposed by this browser';
    const architecture = typeof entropy.architecture === 'string' && entropy.architecture ? String(entropy.architecture) : 'Not exposed by this browser';
    const gpuIdentity = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' · ') || 'Not exposed by this browser';
    const maxBuffer = limits.maxBufferSize ? `${(limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB` : 'Withheld';
    const storageBinding = limits.maxStorageBufferBindingSize ? `${(limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB` : 'Withheld';
    const rows = [
        ['Device class', detectDeviceFamily()],
        ['Exact hardware model', exactModel],
        ['Display profile', displayProfile()],
        ['Browser', detectBrowser()],
        ['Platform', navigator.platform || 'Browser withheld'],
        ['CPU architecture', architecture],
        ['Logical CPU cores', String(nav.hardwareConcurrency || 'Browser withheld')],
        ['Memory hint', nav.deviceMemory ? `${nav.deviceMemory} GB` : 'Browser withheld'],
        ['WebGPU adapter', adapter ? 'Active' : 'Unavailable'],
        ['WebNN accelerator', webnnAvailable ? 'Browser API exposed' : 'Not exposed'],
        ['CPU/WASM fallback', `${FAST_CPU_MODEL_ID} · on demand`],
        ['WASM threads', `${wasmThreads}${self.crossOriginIsolated ? ' · multithread eligible' : ' · single-thread compatibility mode'}`],
        ['GPU identity', gpuIdentity],
        ['GPU max buffer', maxBuffer],
        ['GPU storage binding', storageBinding],
        ['WebGPU features', adapterFeatures.size ? Array.from(adapterFeatures).sort().join(', ') : 'None exposed'],
        ['Storage persistent', storagePersistent ? 'Yes' : 'Browser managed'],
        ['Origin storage used', storageQuota ? `${(storageUsage / 1024 / 1024).toFixed(0)} MB` : 'Unavailable'],
        ['Session persistence', 'Conversation and API key clear on refresh'],
        ['Online', navigator.onLine ? 'Yes' : 'No'],
    ];
    hardware.innerHTML = rows.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd></div>`).join('');
    renderRuntime();
};

const scheduleWarmup = () => {
    const warm = () => {
        if (engine || enginePromise || cpuGenerator || cpuPromise || isGenerating) return;
        if (adapter) void ensureTZ().catch(() => ensureCpu().catch(() => undefined));
        else void ensureCpu().catch(() => undefined);
    };
    const idleWindow = window as Window & { requestIdleCallback?: (callback: () => void, options?: { timeout?: number }) => number };
    if (idleWindow.requestIdleCallback) idleWindow.requestIdleCallback(warm, { timeout: 1400 });
    else setTimeout(warm, 900);
};

const buildConversation = (messages: Msg[], budget: number, systemPrompt: string): ChatMessage[] => {
    const clean = messages.filter(message => !message.pending && message.content.trim());
    const selected: Msg[] = [];
    let used = estimateTokens(systemPrompt);
    for (let index = clean.length - 1; index >= 0; index--) {
        const message = clean[index];
        const tokens = estimateTokens(message.content) + 8;
        if (selected.length >= 2 && used + tokens > budget) break;
        selected.push(message);
        used += tokens;
    }
    selected.reverse();
    const dropped = clean.slice(0, Math.max(0, clean.length - selected.length));
    let memory = '';
    if (dropped.length) {
        const lines = dropped.slice(-8).map(message => `${message.role === 'user' ? 'User' : 'TZ'}: ${message.content.replace(/\s+/g, ' ').slice(0, 180)}`);
        memory = `\nEarlier conversation memory:\n${lines.join('\n')}`.slice(0, 1200);
    }
    const system = `${systemPrompt}${memory}`;
    lastContextMessages = selected.length + (memory ? 1 : 0);
    lastContextTokens = estimateTokens(system) + selected.reduce((sum, message) => sum + estimateTokens(message.content), 0);
    return [{ role: 'system', content: system }, ...selected.map(message => ({ role: message.role, content: message.content }))];
};

const needsLongResponse = (input: string) => /\b(code|script|function|email|letter|write|draft|analy[sz]e|compare|explain|steps|plan|list|report|story|resume)\b/i.test(input) || input.length > 500;

const webgpuGenerate = async (messages: Msg[], onUpdate: (text: string) => void): Promise<GenerationResult> => {
    await ensureTZ();
    const localEngine = engine;
    if (!localEngine) throw new Error(runtimeError || 'WebGPU runtime unavailable');
    const recentUser = [...messages].reverse().find(message => message.role === 'user')?.content || '';
    const localMessages = buildConversation(messages, 2800, tzSystemPrompt);
    const started = performance.now();
    let text = '';
    let first = 0;
    let lastPaint = 0;
    let usage: { prompt_tokens?: number; completion_tokens?: number } | undefined;
    const stream = await localEngine.chat.completions.create({
        messages: localMessages,
        stream: true,
        stream_options: { include_usage: true },
        max_tokens: needsLongResponse(recentUser) ? 768 : 384,
        temperature: 0.25,
        top_p: 0.9,
        repetition_penalty: 1.05,
    });
    for await (const chunk of stream) {
        if (cancelRequested) throw new DOMException('Generation stopped', 'AbortError');
        const delta = chunk.choices[0]?.delta?.content || '';
        if (delta && !first) first = performance.now();
        text += delta;
        if (chunk.usage) usage = chunk.usage;
        const now = performance.now();
        if (delta && now - lastPaint > 70) {
            onUpdate(normalizeOutput(text));
            lastPaint = now;
        }
    }
    text = normalizeOutput(text);
    if (!text) throw new Error('The local model returned an empty response');
    onUpdate(text);
    const elapsed = performance.now() - started;
    const output = usage?.completion_tokens || estimateTokens(text);
    localTps = output / Math.max(0.001, elapsed / 1000);
    localTtft = first ? first - started : elapsed;
    measuredLocalBackend = 'WebGPU / WebLLM worker';
    localThread = 'Dedicated Web Worker';
    try {
        localRuntimeStats = await localEngine.runtimeStatsText();
    } catch {
        localRuntimeStats = 'WebLLM WebGPU runtime';
    }
    renderRuntime();
    return { text, usage, ttft: localTtft, elapsed };
};

const cpuGenerate = async (messages: Msg[], onUpdate: (text: string) => void): Promise<GenerationResult> => {
    await ensureCpu();
    const generator = cpuGenerator;
    if (!generator) throw new Error(runtimeError || 'Low-resource local runtime unavailable');
    const { TextStreamer } = await import('@huggingface/transformers');
    const chat = buildConversation(messages, 1400, cpuSystemPrompt);
    const recentUser = [...messages].reverse().find(message => message.role === 'user')?.content || '';
    const started = performance.now();
    let first = 0;
    let streamedText = '';
    const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (piece: string) => {
            if (!piece || cancelRequested) return;
            if (!first && piece.trim()) first = performance.now();
            streamedText += piece;
            const visible = normalizeOutput(streamedText);
            if (visible) onUpdate(visible);
        },
    });
    const result = await generator(chat, {
        max_new_tokens: needsLongResponse(recentUser) ? 320 : 160,
        do_sample: false,
        repetition_penalty: 1.08,
        return_full_text: false,
        streamer,
    });
    if (cancelRequested) throw new DOMException('Generation stopped', 'AbortError');
    const elapsed = performance.now() - started;
    const text = extractCpuText(result) || normalizeOutput(streamedText);
    if (!text) throw new Error('The low-resource local model returned an empty response');
    onUpdate(text);
    const output = estimateTokens(text);
    localTps = output / Math.max(0.001, elapsed / 1000);
    localTtft = first ? first - started : elapsed;
    localThread = `Transformers.js · ${cpuDeviceLabel} · proxy worker`;
    localRuntimeStats = `Transformers.js ${cpuDeviceLabel} · ${FAST_CPU_MODEL_ID} · ${cpuDtype} · ${wasmThreads} WASM thread${wasmThreads === 1 ? '' : 's'}`;
    engineModelId = FAST_CPU_MODEL_ID;
    measuredLocalBackend = cpuDeviceLabel;
    renderRuntime();
    return { text, ttft: localTtft, elapsed };
};

const localGenerate = async (messages: Msg[], onUpdate: (text: string) => void): Promise<GenerationResult> => {
    let webgpuFailure = '';
    if (adapter) {
        try {
            return await webgpuGenerate(messages, onUpdate);
        } catch (error) {
            if (error instanceof DOMException && error.name === 'AbortError') throw error;
            webgpuFailure = error instanceof Error ? error.message : String(error);
            const failedEngine = engine as (MLCEngineInterface & { unload?: () => Promise<void> }) | null;
            engine = null;
            engineModelId = '';
            engineWorker?.terminate();
            engineWorker = null;
            try { await failedEngine?.unload?.(); } catch { /* best-effort release */ }
        }
    } else {
        webgpuFailure = 'Browser exposed no WebGPU adapter';
    }
    try {
        return await cpuGenerate(messages, onUpdate);
    } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') throw error;
        const cpuFailure = error instanceof Error ? error.message : String(error);
        runtimeError = `WebGPU: ${webgpuFailure} · CPU/WASM: ${cpuFailure}`;
        throw new Error(runtimeError);
    }
};

const chatCompletionsUrl = (endpoint: string) => /\/chat\/completions\/?$/i.test(endpoint)
    ? endpoint.replace(/\/$/, '')
    : `${endpoint.replace(/\/$/, '')}/chat/completions`;

const externalGenerate = async (messages: Msg[], onUpdate: (text: string) => void): Promise<GenerationResult> => {
    if (!apiConfig) throw new Error('No connected AI configured');
    const started = performance.now();
    externalAbort = new AbortController();
    const externalMessages = buildConversation(messages, 14000, tzSystemPrompt);
    const recentUser = [...messages].reverse().find(message => message.role === 'user')?.content || '';
    const response = await fetch(chatCompletionsUrl(apiConfig.endpoint), {
        method: 'POST',
        signal: externalAbort.signal,
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiConfig.key}`,
            'X-Title': 'TZ local-first AI',
        },
        body: JSON.stringify({
            model: apiConfig.model,
            messages: externalMessages,
            stream: true,
            max_tokens: needsLongResponse(recentUser) ? 1400 : 700,
            temperature: 0.25,
        }),
    });
    if (!response.ok) {
        const detail = (await response.text()).slice(0, 240);
        throw new Error(`${apiConfig.provider} returned ${response.status}${detail ? `: ${detail}` : ''}`);
    }
    let text = '';
    let first = 0;
    let usage: { prompt_tokens?: number; completion_tokens?: number } | undefined;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('text/event-stream') && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let lastPaint = 0;
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split(/\r?\n\r?\n/);
            buffer = events.pop() || '';
            for (const event of events) {
                for (const line of event.split(/\r?\n/)) {
                    if (!line.startsWith('data:')) continue;
                    const payload = line.slice(5).trim();
                    if (!payload || payload === '[DONE]') continue;
                    try {
                        const chunk = JSON.parse(payload) as { choices?: Array<{ delta?: { content?: string }; message?: { content?: string } }>; usage?: { prompt_tokens?: number; completion_tokens?: number }; error?: { message?: string } };
                        if (chunk.error?.message) throw new Error(chunk.error.message);
                        const delta = chunk.choices?.[0]?.delta?.content || chunk.choices?.[0]?.message?.content || '';
                        if (delta && !first) first = performance.now();
                        text += delta;
                        if (chunk.usage) usage = chunk.usage;
                        const now = performance.now();
                        if (delta && now - lastPaint > 55) {
                            onUpdate(normalizeOutput(text));
                            lastPaint = now;
                        }
                    } catch (error) {
                        if (error instanceof SyntaxError) continue;
                        throw error;
                    }
                }
            }
        }
    } else {
        const data = await response.json() as { choices?: Array<{ message?: { content?: string } }>; usage?: { prompt_tokens?: number; completion_tokens?: number } };
        text = data.choices?.[0]?.message?.content || '';
        usage = data.usage;
        first = performance.now();
    }
    text = normalizeOutput(text);
    if (!text) throw new Error(`${apiConfig.provider} returned an empty response`);
    onUpdate(text);
    const elapsed = performance.now() - started;
    return { text, usage, ttft: first ? first - started : elapsed, elapsed };
};

const isComplexPrompt = (input: string, messages: Msg[]) => {
    const currentKnowledge = /\b(latest|today|current|news|weather|price|stock|score|schedule|search|look up|verify|202[5-9])\b/i;
    const deeperWork = /\b(research|analy[sz]e|compare|debug|architect|build|code|legal|medical|financial|strategy|business plan|essay|report|reason|investigate)\b/i;
    return currentKnowledge.test(input) || deeperWork.test(input) || input.length > 500 || messages.length > 18;
};

const shouldUseConnectedFirst = (input: string, messages: Msg[]) => Boolean(apiConfig)
    && routingMode !== 'local'
    && (routingMode === 'smart' || isComplexPrompt(input, messages));

const looksWeak = (answer: string, input: string) => {
    const compact = answer.trim();
    if (!compact) return true;
    if (compact.length < 20 && input.length > 80) return true;
    if (/^(?:i(?:'d| would) be happy to assist|how can i help|what's on your mind)/i.test(compact)) return true;
    const words = compact.toLowerCase().match(/[a-z0-9']+/g) || [];
    if (words.length > 18 && new Set(words).size / words.length < 0.38) return true;
    return false;
};

const addInstant = (text: string, result: string) => {
    errorEl.textContent = '';
    history.push({ role: 'user', content: text });
    prompt.value = '';
    const id = turns.length ? Math.max(...turns.map(turn => turn.id)) + 1 : 1;
    turns.push({ id, route: 'Instant', model: 'TZ tools', input: estimateTokens(text), output: estimateTokens(result), ms: 0, tps: 0, ttft: 0 });
    history.push({ role: 'assistant', content: result, turn: id });
    lastRoute = 'Instant';
    renderMessages();
    renderRuntime();
    prompt.focus();
};

const setComposerGenerating = (active: boolean) => {
    isGenerating = active;
    prompt.disabled = active;
    sendBtn.textContent = active ? 'Stop' : 'Send';
    sendBtn.classList.toggle('stop', active);
    sendBtn.setAttribute('aria-label', active ? 'Stop generating' : 'Send message');
};

const stopGeneration = () => {
    cancelRequested = true;
    externalAbort?.abort();
    const interruptible = engine as (MLCEngineInterface & { interruptGenerate?: () => Promise<void> | void }) | null;
    try { void interruptible?.interruptGenerate?.(); } catch { /* best-effort stop */ }
    sendBtn.textContent = 'Stopping…';
};

form.addEventListener('submit', async event => {
    event.preventDefault();
    if (isGenerating) {
        stopGeneration();
        return;
    }
    const text = prompt.value.trim();
    errorEl.textContent = '';
    if (!text) return;
    const instant = instantTool(text);
    if (instant !== null) {
        addInstant(text, instant);
        return;
    }
    void refreshStorage(true);
    cancelRequested = false;
    const runId = ++generationId;
    history.push({ role: 'user', content: text });
    prompt.value = '';
    prompt.style.height = '';
    const id = turns.length ? Math.max(...turns.map(turn => turn.id)) + 1 : 1;
    const assistant: Msg = { role: 'assistant', content: '', turn: id, pending: true };
    history.push(assistant);
    renderMessages();
    setComposerGenerating(true);
    const started = performance.now();
    const context = history.slice(0, -1);
    let result: GenerationResult | null = null;
    let route: Route = 'On-device';
    let model = '';
    const streamUpdate = (partial: string) => {
        if (runId !== generationId || cancelRequested || !partial) return;
        assistant.content = partial;
        assistant.pending = false;
        updateAssistant(assistant);
    };
    try {
        if (shouldUseConnectedFirst(text, context)) {
            try {
                result = await externalGenerate(context, streamUpdate);
                route = 'Connected AI';
                model = apiConfig ? `${apiConfig.provider} / ${apiConfig.model}` : 'Connected AI';
            } catch (externalError) {
                if (externalError instanceof DOMException && externalError.name === 'AbortError') throw externalError;
                assistant.content = '';
                assistant.pending = true;
                renderMessages();
                result = await localGenerate(context, streamUpdate);
                route = 'On-device';
                model = engineModelId || FAST_CPU_MODEL_ID;
            }
        } else {
            try {
                result = await localGenerate(context, streamUpdate);
                route = 'On-device';
                model = engineModelId || FAST_CPU_MODEL_ID;
                if (apiConfig && routingMode === 'auto' && looksWeak(result.text, text)) {
                    assistant.content = '';
                    assistant.pending = true;
                    renderMessages();
                    result = await externalGenerate(context, streamUpdate);
                    route = 'Connected AI';
                    model = `${apiConfig.provider} / ${apiConfig.model}`;
                }
            } catch (localError) {
                if (localError instanceof DOMException && localError.name === 'AbortError') throw localError;
                if (!apiConfig || routingMode === 'local') throw localError;
                assistant.content = '';
                assistant.pending = true;
                renderMessages();
                result = await externalGenerate(context, streamUpdate);
                route = 'Connected AI';
                model = `${apiConfig.provider} / ${apiConfig.model}`;
            }
        }
        if (!result || runId !== generationId) return;
        assistant.content = result.text;
        assistant.pending = false;
        lastRoute = route;
        const elapsed = performance.now() - started;
        const input = result.usage?.prompt_tokens || lastContextTokens || estimateTokens(context.map(message => message.content).join('\n'));
        const output = result.usage?.completion_tokens || estimateTokens(result.text);
        const tps = route === 'On-device' ? localTps : output / Math.max(0.001, result.elapsed / 1000);
        turns.push({ id, route, model, input, output, ms: elapsed, tps, ttft: result.ttft });
        renderMessages();
        renderRuntime();
    } catch (error) {
        if (runId !== generationId) return;
        const stopped = error instanceof DOMException && error.name === 'AbortError';
        if (stopped) {
            if (!assistant.content) history.pop();
            else {
                assistant.content = normalizeOutput(assistant.content);
                assistant.pending = false;
            }
        } else {
            history.pop();
            runtimeError = error instanceof Error ? error.message : String(error);
            errorEl.textContent = apiConfig
                ? 'TZ could not complete this turn locally or through the connected service. Check Monitoring for the exact error and retry.'
                : 'The local engine could not complete this request. Open Monitoring to retry, or connect a free-tier provider for difficult prompts.';
        }
        renderMessages();
        renderRuntime();
    } finally {
        externalAbort = null;
        if (runId === generationId) {
            setComposerGenerating(false);
            prompt.focus();
        }
    }
});

const clearSession = () => {
    if (isGenerating) stopGeneration();
    generationId++;
    history = [];
    turns = [];
    lastRoute = 'Instant';
    lastContextMessages = 0;
    lastContextTokens = 0;
    prompt.value = '';
    prompt.style.height = '';
    errorEl.textContent = '';
    setComposerGenerating(false);
    renderMessages();
    renderRuntime();
};

['clearBtn', 'clearCacheBtn'].forEach(id => {
    $<HTMLButtonElement>(id).addEventListener('click', event => {
        event.preventDefault();
        event.stopPropagation();
        clearSession();
    });
});

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
        `Selected model: ${selectedModelRecord?.model_id || 'none'}`,
        `Loaded model: ${engineModelId || 'none'}`,
        `Execution backend: ${measuredLocalBackend || 'not measured'}`,
        `Routing mode: ${modeLabel()}`,
        `Connected service: ${apiConfig ? `${apiConfig.provider} / ${apiConfig.model}` : 'none'}`,
        `Cross-origin isolated: ${self.crossOriginIsolated}`,
        `Logical cores: ${nav.hardwareConcurrency || 'withheld'}`,
        `Memory hint: ${nav.deviceMemory ? `${nav.deviceMemory} GB` : 'withheld'}`,
        `Last local error: ${runtimeError || 'none'}`,
    ].join('\n');
};

const retryLocalEngine = async () => {
    const retryButton = $<HTMLButtonElement>('retryRuntimeBtn');
    const diagnosticStatus = $<HTMLElement>('diagnosticStatus');
    if (enginePromise || cpuPromise) {
        diagnosticStatus.textContent = 'TZ is still loading. Current progress is shown above.';
        return;
    }
    retryButton.disabled = true;
    diagnosticStatus.textContent = 'Restarting the local engine…';
    try {
        const previousEngine = engine as (MLCEngineInterface & { unload?: () => Promise<void> }) | null;
        engine = null;
        engineModelId = '';
        engineWorker?.terminate();
        engineWorker = null;
        try { await previousEngine?.unload?.(); } catch { /* best-effort GPU release */ }
        const previousCpu = cpuGenerator;
        cpuGenerator = null;
        try { await previousCpu?.dispose?.(); } catch { /* best-effort WASM release */ }
        runtimeError = '';
        loadProgress = 0;
        cpuLoadProgress = 0;
        await loadHardware();
        if (adapter) await ensureTZ();
        if (!engine) await ensureCpu();
        diagnosticStatus.textContent = engine ? 'Quality WebGPU engine ready.' : 'Low-resource CPU/WASM engine ready.';
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
    $<HTMLElement>('diagnosticStatus').textContent = 'Diagnostics copied. Prompt text and API keys are excluded.';
});

const applyPreset = () => {
    const preset = PROVIDERS[apiPreset.value] || PROVIDERS.custom;
    if (apiPreset.value !== 'custom') {
        apiEndpoint.value = preset.endpoint;
        apiModel.value = preset.model;
    }
    apiStatus.textContent = preset.note;
};

const syncApiPanel = () => {
    connectBtn.setAttribute('aria-expanded', String(!connectPanel.hidden));
    disconnectApiBtn.disabled = !apiConfig;
    routeMode.value = routingMode;
    if (apiConfig) {
        apiStatus.textContent = `${apiConfig.provider} connected for this tab. ${modeLabel()} is active; the key clears on refresh.`;
    }
    $<HTMLElement>('apiKeyAudit').textContent = apiConfig ? `${apiConfig.provider} key in tab memory · refresh clears it` : 'None';
    renderRuntime();
};

connectBtn.addEventListener('click', () => {
    connectPanel.hidden = !connectPanel.hidden;
    syncApiPanel();
    if (!connectPanel.hidden) apiPreset.focus();
});

apiPreset.addEventListener('change', applyPreset);
routeMode.addEventListener('change', () => {
    routingMode = routeMode.value as RoutingMode;
    syncApiPanel();
});

apiEndpoint.addEventListener('input', () => {
    const value = apiEndpoint.value.trim();
    if (/^http:\/\//i.test(value)) {
        apiStatus.textContent = 'Connected AI endpoints must use HTTPS.';
        apiEndpoint.setCustomValidity('Connected AI endpoints must use HTTPS.');
    } else {
        apiEndpoint.setCustomValidity('');
    }
});

apiForm.addEventListener('submit', event => {
    event.preventDefault();
    const endpoint = apiEndpoint.value.trim().replace(/\/$/, '');
    const model = apiModel.value.trim();
    const key = apiKey.value.trim();
    if (!endpoint || !model || !key) {
        apiStatus.textContent = 'Provider, model, and API key are required.';
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
        apiStatus.textContent = 'Connected AI endpoints must use HTTPS.';
        return;
    }
    const preset = PROVIDERS[apiPreset.value] || PROVIDERS.custom;
    apiConfig = { provider: preset.name, endpoint, model, key };
    routingMode = routeMode.value as RoutingMode;
    apiKey.value = '';
    syncApiPanel();
});

disconnectApiBtn.addEventListener('click', () => {
    apiConfig = null;
    apiStatus.textContent = 'Connected AI removed. TZ is local-only until another provider is connected.';
    syncApiPanel();
});

prompt.addEventListener('input', () => {
    prompt.style.height = 'auto';
    prompt.style.height = `${Math.min(180, prompt.scrollHeight)}px`;
});

prompt.addEventListener('keydown', event => {
    const desktopEnter = event.key === 'Enter' && !event.shiftKey && !isMobileDevice();
    const shortcutEnter = event.key === 'Enter' && (event.metaKey || event.ctrlKey);
    if (desktopEnter || shortcutEnter) {
        event.preventDefault();
        form.requestSubmit();
    }
});

const navButtons = [...document.querySelectorAll<HTMLButtonElement>('.nav')];
const menuBtn = document.getElementById('menuBtn') as HTMLButtonElement;
const primaryNav = document.getElementById('primaryNav') as HTMLElement;
const selectView = (button: HTMLButtonElement) => {
    navButtons.forEach(item => item.classList.toggle('active', item === button));
    document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
    document.getElementById(`${button.dataset.view}View`)?.classList.add('active');
    primaryNav.classList.remove('open');
    menuBtn.setAttribute('aria-expanded', 'false');
    history.replaceState(null, '', button.dataset.view === 'home' ? location.pathname : `#${button.dataset.view}`);
    window.scrollTo({ top: 0, behavior: 'auto' });
    if (button.dataset.view === 'chat' && !engine && !enginePromise && !cpuGenerator && !cpuPromise) void loadHardware().then(scheduleWarmup);
};
navButtons.forEach(button => button.addEventListener('click', () => selectView(button)));
document.querySelectorAll<HTMLElement>('[data-open-view]').forEach(control => control.addEventListener('click', () => {
    const target = navButtons.find(button => button.dataset.view === control.dataset.openView);
    if (target) selectView(target);
}));
menuBtn.addEventListener('click', () => {
    const open = primaryNav.classList.toggle('open');
    menuBtn.setAttribute('aria-expanded', String(open));
});

const animationFiles = [
    '1-1-1.html', '10101010101011.html', '111-11221.html', 'AASZZ.html', 'FPS AIM TRAINER.html',
    'MOBILE ANIMATION.html', 'SERIAL EXPERIMENT LAIN - MOBILE WEBCAM PREVIEW.html', 'alphayo.html',
    'character_generator.html', 'connected lines and sound wave.html', 'connections-animation.html',
    'data tracking v1.1.html', 'facial-tracking-p5.html', 'fake login screen.html', 'fighter-jet-hud.html',
    'hand animation using geo-nodes and lines.html', 'multiple animations and custom image loader.html',
    'pixel liquid.html', 'tactical-military-ui.html', 'webcam preview - serial experiment lain inspiration.html',
    'windows 7 webcam preview.html', 'wpm game.html',
];
const projectTitle = (file: string) => file.replace(/\.html$/i, '').replace(/[-_]/g, ' ').replace(/\b\w/g, value => value.toUpperCase());
const animationGrid = document.getElementById('animationGrid');
if (animationGrid) animationGrid.innerHTML = animationFiles.map((file, index) => {
    const url = `/projects/html-animations/${encodeURIComponent(file)}`;
    return `<article class='project-card'><a class='project-preview' href='${url}' target='_blank' rel='noreferrer' aria-label='Open ${escapeHtml(projectTitle(file))}'><iframe src='${url}' title='' tabindex='-1' loading='lazy' sandbox='allow-scripts'></iframe><span>Open experiment ↗</span></a><div class='project-meta'><small>${String(index + 1).padStart(2, '0')}</small><h3>${escapeHtml(projectTitle(file))}</h3></div></article>`;
}).join('');

const initialView = location.hash === '#portfolio' ? 'portfolio' : location.hash === '#monitor' ? 'monitor' : location.hash === '#chat' ? 'chat' : 'home';
const initialButton = navButtons.find(button => button.dataset.view === initialView);
if (initialButton) selectView(initialButton);

applyPreset();
syncApiPanel();
renderMessages();
renderTelemetry();
renderRuntimeNow();
if (initialView === 'chat') void loadHardware().then(scheduleWarmup);
else void loadHardware();
addEventListener('online', () => document.getElementById('chatView')?.classList.contains('active') ? void loadHardware().then(scheduleWarmup) : void loadHardware());
addEventListener('pageshow', event => {
    if ((event as PageTransitionEvent).persisted) {
        apiConfig = null;
        routingMode = 'auto';
        clearSession();
        syncApiPanel();
    }
    void loadHardware().then(scheduleWarmup);
});
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && !engine && !enginePromise) void loadHardware().then(scheduleWarmup);
});
