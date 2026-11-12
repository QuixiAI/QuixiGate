import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useChat, type UIMessage } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import {
  Brain,
  ChevronDown,
  ChevronRight,
  Columns2,
  Globe,
  MessageSquare,
  PencilLine,
  PlusCircle,
  RotateCcw,
  Send,
  SlidersHorizontal,
  Square,
  Trash2,
  Wrench,
} from 'lucide-react';
import { api, isModelRoutable } from '../api';
import { MarkdownMessage } from '../components/MarkdownMessage';
import type {
  CatalogModel,
  ChatMessageMetadata,
  CredentialRecord,
  ModelRoute,
} from '../types';

type PlaygroundMessage = UIMessage<ChatMessageMetadata>;

const REASONING_OPEN_STORAGE_KEY = 'quixigate.playground.reasoningOpen';
const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 24;

function readInitialReasoningOpen(): boolean {
  try {
    const val = window.localStorage.getItem(REASONING_OPEN_STORAGE_KEY);
    return val !== 'false';
  } catch {
    return true;
  }
}

function writeReasoningOpenPreference(open: boolean) {
  try {
    window.localStorage.setItem(REASONING_OPEN_STORAGE_KEY, String(open));
  } catch {
    // Ignore storage failures in restricted contexts
  }
}

function promptRequestsWebSearch(prompt: string): boolean {
  return /\b(search\s+(?:the\s+)?(?:web|internet|online)|browse\s+(?:the\s+)?(?:web|internet|online)|look\s+up\s+online|google\s+(?:for|about)|find\s+(?:recent|latest)\s+(?:articles|news)|latest\s+(?:news|articles)\s+(?:on|about))\b|https?:\/\//i.test(
    prompt,
  );
}

function replyDeclinesWebAccess(reply: string): boolean {
  return /\b(don['’]?t\s+have\s+(?:the\s+)?ability\s+to\s+(?:search|browse)|can(?:not|['’]?t)\s+(?:search|browse|access)\s+(?:the\s+)?(?:web|internet|online)|unable\s+to\s+(?:search|browse|access)\s+(?:the\s+)?(?:web|internet)|don['’]?t\s+have\s+(?:browsing|internet|web)\s+(?:capabilities|access))\b/i.test(
    reply,
  );
}

function formatPlaygroundError(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.startsWith('{')) {
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (parsed && typeof parsed === 'object') {
        const obj = parsed as Record<string, unknown>;
        if (obj.error && typeof obj.error === 'object') {
          const errObj = obj.error as Record<string, unknown>;
          if (typeof errObj.message === 'string' && errObj.message.trim()) {
            return errObj.message;
          }
        }
        if (typeof obj.message === 'string' && obj.message.trim()) {
          return obj.message;
        }
      }
    } catch {
      // Fall back to raw message
    }
  }
  return raw;
}

interface ChatPaneProps {
  paneLabel?: string;
  selectedModel: string;
  onSelectModel: (m: string) => void;
  modelOptions: string[];
  systemPrompt: string;
  temperature: number;
  reasoningEffort: string;
  onlineSearch: boolean;
  onEnableOnlineSearch: () => void;
  routingModifier?: 'none' | 'nitro' | 'floor';
  broadcastSignal: { seq: number; text: string } | null;
  hideComposer?: boolean;
  onNavigateProviders?: () => void;
}

function ChatPane({
  paneLabel,
  selectedModel,
  onSelectModel,
  modelOptions,
  systemPrompt,
  temperature,
  reasoningEffort,
  onlineSearch,
  onEnableOnlineSearch,
  routingModifier = 'none',
  broadcastSignal,
  hideComposer,
  onNavigateProviders,
}: ChatPaneProps) {
  const effectiveModel =
    routingModifier !== 'none' &&
    selectedModel &&
    !selectedModel.includes(':nitro') &&
    !selectedModel.includes(':floor')
      ? `${selectedModel}:${routingModifier}`
      : selectedModel;
  const chatId = useId();
  const [customSlugMode, setCustomSlugMode] = useState(false);
  const transport = useMemo(
    () => new DefaultChatTransport<PlaygroundMessage>({ api: '/api/chat' }),
    [],
  );

  const { messages, sendMessage, stop, setMessages, status, error, clearError } =
    useChat<PlaygroundMessage>({
      id: chatId,
      transport,
    });

  const [localInput, setLocalInput] = useState('');
  const [sessionSeedOpen, setSessionSeedOpen] = useState<boolean>(
    readInitialReasoningOpen,
  );
  const [expandedReasoning, setExpandedReasoning] = useState<Record<string, boolean>>({});
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const [lastBroadcastSeq, setLastBroadcastSeq] = useState(0);

  const feedRef = useRef<HTMLDivElement | null>(null);
  const feedContentRef = useRef<HTMLDivElement | null>(null);
  const isPinnedToBottomRef = useRef(true);
  const pendingSubmitScrollRef = useRef(false);
  const wasStreamingRef = useRef(false);
  const isStreamingRef = useRef(false);
  const lastScrollTopRef = useRef(0);

  const isStreaming = status === 'submitted' || status === 'streaming';
  isStreamingRef.current = isStreaming;

  // Inherit default open/collapsed state from the preceding (current) thought trace
  const { resolvedReasoningOpen, latestReasoningKey } = useMemo(() => {
    const resolved: Record<string, boolean> = {};
    let prevOpen = sessionSeedOpen;
    let lastKey: string | null = null;
    for (const msg of messages) {
      msg.parts.forEach((part, idx) => {
        if (part.type === 'reasoning') {
          const key = `${msg.id}-${idx}`;
          const isOpen = expandedReasoning[key] ?? prevOpen;
          resolved[key] = isOpen;
          prevOpen = isOpen;
          lastKey = key;
        }
      });
    }
    return { resolvedReasoningOpen: resolved, latestReasoningKey: lastKey };
  }, [messages, expandedReasoning, sessionSeedOpen]);

  // Pin each newly seen reasoning trace's initial state so toggling an older trace
  // never retroactively flips newer traces
  useEffect(() => {
    const missingEntries: Array<[string, boolean]> = [];
    for (const [key, isOpen] of Object.entries(resolvedReasoningOpen)) {
      if (!(key in expandedReasoning)) {
        missingEntries.push([key, isOpen]);
      }
    }
    if (missingEntries.length > 0) {
      setExpandedReasoning((prev) => {
        const next = { ...prev };
        for (const [key, isOpen] of missingEntries) {
          if (!(key in next)) {
            next[key] = isOpen;
          }
        }
        return next;
      });
    }
  }, [resolvedReasoningOpen, expandedReasoning]);

  const scrollFeedToBottom = useCallback(() => {
    const el = feedRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    lastScrollTopRef.current = el.scrollTop;
  }, []);

  const engageAutoScrollForSubmit = useCallback(() => {
    isPinnedToBottomRef.current = true;
    pendingSubmitScrollRef.current = true;
    requestAnimationFrame(() => {
      if (isPinnedToBottomRef.current) {
        scrollFeedToBottom();
      }
    });
  }, [scrollFeedToBottom]);

  function handleFeedWheel(e: React.WheelEvent<HTMLDivElement>) {
    if (e.deltaY < 0) {
      // Immediately pause auto-scrolling when the user wheels/trackpads upward
      isPinnedToBottomRef.current = false;
      pendingSubmitScrollRef.current = false;
    } else if (e.deltaY > 0 && feedRef.current) {
      const el = feedRef.current;
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (distanceFromBottom <= AUTO_SCROLL_BOTTOM_THRESHOLD_PX) {
        isPinnedToBottomRef.current = true;
      }
    }
  }

  function handleFeedScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    const currentTop = el.scrollTop;
    const prevTop = lastScrollTopRef.current;
    const distanceFromBottom = el.scrollHeight - currentTop - el.clientHeight;

    if (
      currentTop < prevTop - 1 &&
      distanceFromBottom > AUTO_SCROLL_BOTTOM_THRESHOLD_PX
    ) {
      // User scrolled up away from the bottom -> pause auto-scrolling
      isPinnedToBottomRef.current = false;
      pendingSubmitScrollRef.current = false;
    } else if (
      distanceFromBottom <= AUTO_SCROLL_BOTTOM_THRESHOLD_PX &&
      currentTop >= prevTop
    ) {
      // User scrolled back to the bottom -> resume auto-scrolling
      isPinnedToBottomRef.current = true;
    }

    lastScrollTopRef.current = currentTop;
  }

  // Auto-scroll synchronously before paint as messages stream in (unless user scrolled up)
  useLayoutEffect(() => {
    const shouldAutoScroll =
      isPinnedToBottomRef.current &&
      (isStreaming || wasStreamingRef.current || pendingSubmitScrollRef.current);

    if (shouldAutoScroll) {
      scrollFeedToBottom();
      if (isStreaming) {
        pendingSubmitScrollRef.current = false;
      }
    }

    if (!isStreaming && wasStreamingRef.current) {
      pendingSubmitScrollRef.current = false;
    }
    wasStreamingRef.current = isStreaming;
  }, [messages, isStreaming, error, scrollFeedToBottom]);

  // Keep pinned when inner content resizes during an active stream
  useEffect(() => {
    const contentEl = feedContentRef.current;
    if (!contentEl || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => {
      if (
        isPinnedToBottomRef.current &&
        (isStreamingRef.current || pendingSubmitScrollRef.current)
      ) {
        scrollFeedToBottom();
      }
    });
    observer.observe(contentEl);
    return () => observer.disconnect();
  }, [messages.length, scrollFeedToBottom]);

  // Trigger simultaneous Arena dispatch when broadcastSignal increments
  useEffect(() => {
    if (
      broadcastSignal &&
      broadcastSignal.seq > lastBroadcastSeq &&
      broadcastSignal.text.trim()
    ) {
      setLastBroadcastSeq(broadcastSignal.seq);
      engageAutoScrollForSubmit();
      void sendMessage(
        { text: broadcastSignal.text },
        {
          body: {
            model: effectiveModel,
            system: systemPrompt || undefined,
            temperature,
            reasoningEffort: reasoningEffort === 'default' ? undefined : reasoningEffort,
            online: onlineSearch,
          },
        },
      );
    }
  }, [
    broadcastSignal,
    effectiveModel,
    engageAutoScrollForSubmit,
    lastBroadcastSeq,
    onlineSearch,
    reasoningEffort,
    sendMessage,
    systemPrompt,
    temperature,
  ]);

  const onlineSearchRetryPrompt = useMemo(() => {
    if (onlineSearch || isStreaming || messages.length === 0) {
      return null;
    }
    let lastUserText = '';
    let lastAssistantText = '';
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (!lastAssistantText && m.role === 'assistant') {
        lastAssistantText = m.parts
          .filter((p) => p.type === 'text')
          .map((p) => p.text)
          .join('\n');
      }
      if (!lastUserText && m.role === 'user') {
        lastUserText = m.parts
          .filter((p) => p.type === 'text')
          .map((p) => p.text)
          .join('\n')
          .trim();
        break;
      }
    }
    if (!lastUserText) return null;
    if (
      promptRequestsWebSearch(lastUserText) ||
      replyDeclinesWebAccess(lastAssistantText)
    ) {
      return lastUserText;
    }
    return null;
  }, [onlineSearch, isStreaming, messages]);

  function handleEnableOnlineAndRetry() {
    if (!onlineSearchRetryPrompt || isStreaming) return;
    onEnableOnlineSearch();
    engageAutoScrollForSubmit();
    void sendMessage(
      { text: onlineSearchRetryPrompt },
      {
        body: {
          model: effectiveModel,
          system: systemPrompt || undefined,
          temperature,
          reasoningEffort: reasoningEffort === 'default' ? undefined : reasoningEffort,
          online: true,
        },
      },
    );
  }

  function handleLocalSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = localInput.trim();
    if (!trimmed || isStreaming) return;
    setLocalInput('');
    engageAutoScrollForSubmit();
    void sendMessage(
      { text: trimmed },
      {
        body: {
          model: effectiveModel,
          system: systemPrompt || undefined,
          temperature,
          reasoningEffort: reasoningEffort === 'default' ? undefined : reasoningEffort,
          online: onlineSearch,
        },
      },
    );
  }

  return (
    <div className="flex flex-col h-full rounded-2xl bg-[#0d111a] border border-white/8 overflow-hidden">
      {/* Pane Top Bar */}
      <div className="px-4 py-3 border-b border-white/8 bg-slate-950/60 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-1 min-w-[220px]">
          {paneLabel && (
            <span className="rounded bg-indigo-500/20 border border-indigo-500/30 px-2 py-0.5 font-mono text-[11px] font-semibold text-indigo-300">
              {paneLabel}
            </span>
          )}
          {modelOptions.length > 0 && !customSlugMode ? (
            <select
              value={
                modelOptions.includes(selectedModel)
                  ? selectedModel
                  : modelOptions[0]
              }
              onChange={(e) => onSelectModel(e.target.value)}
              className="flex-1 rounded-lg bg-slate-900 border border-white/10 px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-indigo-400"
            >
              {modelOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : (
            <input
              list={`models-${chatId}`}
              value={selectedModel}
              onChange={(e) => onSelectModel(e.target.value)}
              placeholder={
                modelOptions.length === 0
                  ? 'Connect a provider seat or start Ollama first…'
                  : 'author/model@provider:online or alias'
              }
              className="flex-1 rounded-lg bg-slate-900 border border-white/10 px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-indigo-400"
            />
          )}
          <datalist id={`models-${chatId}`}>
            {modelOptions.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
          {modelOptions.length > 0 && (
            <button
              type="button"
              onClick={() => setCustomSlugMode((v) => !v)}
              title={
                customSlugMode
                  ? 'Switch back to configured model dropdown'
                  : 'Type custom model@provider:online slug'
              }
              className={`rounded-lg px-2 py-1.5 text-[11px] border cursor-pointer inline-flex items-center gap-1 ${
                customSlugMode
                  ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-200'
                  : 'bg-white/5 hover:bg-white/10 border-white/10 text-slate-300'
              }`}
            >
              <PencilLine className="h-3 w-3" />
              {customSlugMode ? 'Dropdown' : 'Custom'}
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={() => {
            if (latestReasoningKey) {
              const lastOpen = resolvedReasoningOpen[latestReasoningKey];
              setSessionSeedOpen(lastOpen);
              writeReasoningOpenPreference(lastOpen);
            }
            clearError();
            setExpandedReasoning({});
            setExpandedTools({});
            isPinnedToBottomRef.current = true;
            lastScrollTopRef.current = 0;
            setMessages([]);
          }}
          title="Clear conversation"
          className="rounded-lg bg-white/5 hover:bg-white/10 p-1.5 text-slate-400 hover:text-white cursor-pointer"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Messages Feed */}
      <div
        ref={feedRef}
        data-testid="chat-messages-feed"
        onScroll={handleFeedScroll}
        onWheel={handleFeedWheel}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {messages.length === 0 ? (
          modelOptions.length === 0 && !selectedModel ? (
            <div className="h-full flex flex-col items-center justify-center text-center py-12 px-4">
              <div className="h-11 w-11 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center text-amber-300 mb-3">
                <PlusCircle className="h-5 w-5" />
              </div>
              <p className="text-sm font-semibold text-white">
                No Providers or Local Runtimes Connected Yet
              </p>
              <p className="text-xs text-slate-400 mt-1.5 max-w-md">
                Playground only lists models backed by your active OAuth seats, API keys, or running local servers (Ollama, LM Studio, llama.cpp, vLLM).
              </p>
              {onNavigateProviders && (
                <button
                  type="button"
                  onClick={onNavigateProviders}
                  className="mt-4 inline-flex items-center gap-2 rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-xs font-semibold text-white cursor-pointer"
                >
                  <PlusCircle className="h-3.5 w-3.5" />
                  Connect Provider or OAuth Seat
                </button>
              )}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center py-12 px-4">
              <div className="h-10 w-10 rounded-xl bg-indigo-500/15 border border-indigo-500/25 flex items-center justify-center text-indigo-400 mb-3">
                <MessageSquare className="h-5 w-5" />
              </div>
              <p className="text-sm font-medium text-slate-200">
                Ready to stream via <code className="text-indigo-300">{selectedModel}</code>
              </p>
              <p className="text-xs text-slate-400 mt-1 max-w-sm">
                Powered by <code className="text-slate-300">@ai-sdk/react</code> UIMessageStream with live reasoning blocks, tool execution, and exact NanoUsd cost telemetry.
              </p>
            </div>
          )
        ) : (
          <div ref={feedContentRef} className="space-y-4">
            {messages.map((msg) => {
            const isUser = msg.role === 'user';
            const meta = msg.metadata;

            return (
              <div
                key={msg.id}
                className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[90%] rounded-2xl px-4 py-3 text-sm space-y-2.5 ${
                    isUser
                      ? 'bg-indigo-500 text-white'
                      : 'bg-slate-950/90 border border-white/8 text-slate-100'
                  }`}
                >
                  {msg.parts.map((part, idx) => {
                    const partKey = `${msg.id}-${idx}`;
                    if (part.type === 'reasoning') {
                      const isOpen =
                        resolvedReasoningOpen[partKey] ?? sessionSeedOpen;
                      return (
                        <div
                          key={partKey}
                          className="rounded-xl bg-amber-500/8 border border-amber-500/25 p-2.5 text-xs"
                        >
                          <button
                            type="button"
                            aria-expanded={isOpen}
                            onClick={() => {
                              const nextOpen = !isOpen;
                              setExpandedReasoning((prev) => ({
                                ...prev,
                                [partKey]: nextOpen,
                              }));
                              if (partKey === latestReasoningKey) {
                                setSessionSeedOpen(nextOpen);
                                writeReasoningOpenPreference(nextOpen);
                              }
                            }}
                            className="flex items-center gap-1.5 font-medium text-amber-300 cursor-pointer w-full text-left"
                          >
                            {isOpen ? (
                              <ChevronDown className="h-3.5 w-3.5" />
                            ) : (
                              <ChevronRight className="h-3.5 w-3.5" />
                            )}
                            <Brain className="h-3.5 w-3.5" />
                            Reasoning Trace
                          </button>
                          {isOpen && (
                            <div className="mt-2 font-mono text-[11px] text-amber-100/85 whitespace-pre-wrap leading-relaxed">
                              {part.text}
                            </div>
                          )}
                        </div>
                      );
                    }

                    if (part.type === 'text') {
                      if (isUser) {
                        return (
                          <div
                            key={partKey}
                            className="whitespace-pre-wrap leading-relaxed"
                          >
                            {part.text}
                          </div>
                        );
                      }
                      return <MarkdownMessage key={partKey} text={part.text} />;
                    }

                    if (part.type.startsWith('tool-') || part.type === 'dynamic-tool') {
                      const toolPart = part as unknown as {
                        toolName?: string;
                        toolCallId?: string;
                        input?: unknown;
                        output?: unknown;
                        state?: string;
                      };
                      const resolvedName =
                        toolPart.toolName ?? part.type.replace(/^tool-/, '');
                      const parsedInput =
                        toolPart.input && typeof toolPart.input === 'object'
                          ? (toolPart.input as Record<string, unknown>)
                          : null;
                      let parsedOutput: Record<string, unknown> | null = null;
                      if (typeof toolPart.output === 'string') {
                        try {
                          const candidate: unknown = JSON.parse(toolPart.output);
                          if (candidate && typeof candidate === 'object') {
                            parsedOutput = candidate as Record<string, unknown>;
                          }
                        } catch {
                          parsedOutput = null;
                        }
                      } else if (toolPart.output && typeof toolPart.output === 'object') {
                        parsedOutput = toolPart.output as Record<string, unknown>;
                      }

                      const providerObj =
                        parsedOutput?.provider && typeof parsedOutput.provider === 'object'
                          ? (parsedOutput.provider as Record<string, unknown>)
                          : null;
                      const providerLabel =
                        (typeof providerObj?.label === 'string' && providerObj.label) ||
                        (typeof parsedOutput?.provider === 'string' && parsedOutput.provider) ||
                        null;
                      const searchHits = Array.isArray(parsedOutput?.results)
                        ? (parsedOutput.results as Array<Record<string, unknown>>)
                        : null;
                      const queryLabel =
                        (typeof parsedInput?.query === 'string' && parsedInput.query) ||
                        (typeof parsedInput?.url === 'string' && parsedInput.url) ||
                        null;
                      const isToolOpen = expandedTools[partKey] ?? false;

                      return (
                        <div
                          key={partKey}
                          className="rounded-xl bg-indigo-500/10 border border-indigo-500/25 p-2.5 text-xs font-mono space-y-1.5"
                        >
                          <button
                            type="button"
                            onClick={() =>
                              setExpandedTools((prev) => ({
                                ...prev,
                                [partKey]: !isToolOpen,
                              }))
                            }
                            className="flex flex-wrap items-center justify-between gap-2 w-full text-left cursor-pointer"
                          >
                            <span className="flex items-center gap-1.5 text-indigo-300 font-semibold">
                              {isToolOpen ? (
                                <ChevronDown className="h-3.5 w-3.5 shrink-0" />
                              ) : (
                                <ChevronRight className="h-3.5 w-3.5 shrink-0" />
                              )}
                              <Wrench className="h-3.5 w-3.5 shrink-0" />
                              <span>{resolvedName}</span>
                              {queryLabel && (
                                <span className="text-slate-300 font-normal truncate max-w-[320px]">
                                  — &ldquo;{queryLabel}&rdquo;
                                </span>
                              )}
                            </span>
                            <span className="flex items-center gap-1.5 text-[10px]">
                              {providerLabel && (
                                <span className="rounded bg-emerald-500/15 border border-emerald-500/30 px-1.5 py-0.5 text-emerald-300">
                                  {providerLabel}
                                </span>
                              )}
                              {searchHits && (
                                <span className="text-slate-400">
                                  {searchHits.length} hits
                                </span>
                              )}
                              {typeof parsedOutput?.took_ms === 'number' && (
                                <span className="text-slate-400">
                                  · {parsedOutput.took_ms}ms
                                </span>
                              )}
                              {parsedOutput?.cached === true && (
                                <span className="rounded bg-sky-500/15 px-1.5 py-0.5 text-sky-300">
                                  CACHED
                                </span>
                              )}
                            </span>
                          </button>

                          {!isToolOpen && searchHits && searchHits.length > 0 && (
                            <div className="pt-1 border-t border-white/10 flex flex-wrap gap-1.5">
                              {searchHits.slice(0, 4).map((hit, idx) => {
                                const url = typeof hit.url === 'string' ? hit.url : '';
                                const title =
                                  (typeof hit.title === 'string' && hit.title) ||
                                  (typeof hit.site_name === 'string' && hit.site_name) ||
                                  url;
                                if (!url) return null;
                                return (
                                  <a
                                    key={`${url}-${idx}`}
                                    href={url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="rounded-md bg-slate-950/70 border border-white/10 hover:border-indigo-400/50 px-2 py-0.5 text-[11px] text-slate-300 hover:text-white truncate max-w-[260px]"
                                  >
                                    {title}
                                  </a>
                                );
                              })}
                            </div>
                          )}

                          {isToolOpen && (
                            <div className="space-y-1.5 pt-1 border-t border-white/10">
                              {toolPart.input !== undefined && (
                                <pre className="text-[11px] text-slate-300 overflow-x-auto">
                                  {JSON.stringify(toolPart.input, null, 2)}
                                </pre>
                              )}
                              {toolPart.output !== undefined && (
                                <div className="pt-1 border-t border-white/10 text-[11px] text-emerald-300 whitespace-pre-wrap max-h-64 overflow-y-auto">
                                  {typeof toolPart.output === 'string'
                                    ? toolPart.output
                                    : JSON.stringify(toolPart.output, null, 2)}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    }

                    return null;
                  })}
                </div>

                {!isUser && meta && (
                  <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[11px] font-mono text-slate-400 px-1">
                    {meta.provider && (
                      <span className="text-indigo-300">{meta.provider}</span>
                    )}
                    {meta.model && <span>· {meta.model}</span>}
                    {meta.ttftMs != null && <span>· TTFT {meta.ttftMs}ms</span>}
                    {meta.durationMs != null && <span>· {meta.durationMs}ms</span>}
                    {meta.outputTokens != null && (
                      <span>
                        · {(meta.inputTokens ?? 0) + meta.outputTokens} tok
                      </span>
                    )}
                    {meta.costFormatted && (
                      <span className="text-slate-200">· {meta.costFormatted}</span>
                    )}
                    {meta.costNanos === 0 && (meta.apiEquivNanos ?? 0) > 0 && (
                      <span className="text-emerald-400">
                        (saved {meta.apiEquivFormatted})
                      </span>
                    )}
                    {meta.cacheHit && (
                      <span className="rounded bg-sky-500/15 px-1.5 py-0.5 text-[10px] text-sky-300">
                        CACHE HIT
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          </div>
        )}

        {onlineSearchRetryPrompt && (
          <div className="rounded-xl bg-amber-500/10 border border-amber-500/30 px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className="inline-flex items-center gap-2 text-amber-200">
              <Globe className="h-3.5 w-3.5 text-amber-300 shrink-0" />
              <span>
                Web search (<code className="font-mono text-amber-100">:online</code>) was turned off for this turn.
              </span>
            </span>
            <button
              type="button"
              onClick={handleEnableOnlineAndRetry}
              className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-400/40 px-2.5 py-1 font-medium text-emerald-200 cursor-pointer"
            >
              <RotateCcw className="h-3 w-3" />
              Enable :online search &amp; retry
            </button>
          </div>
        )}

        {error && (
          <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 p-3 text-xs text-rose-200 whitespace-pre-wrap">
            {formatPlaygroundError(error.message)}
          </div>
        )}
      </div>

      {/* Single-Pane Composer */}
      {!hideComposer && (
        <form
          onSubmit={handleLocalSubmit}
          className="p-3 border-t border-white/8 bg-slate-950/60 space-y-2"
        >
          {!onlineSearch && promptRequestsWebSearch(localInput) && (
            <div className="flex items-center justify-between gap-2 rounded-lg bg-amber-500/10 border border-amber-500/30 px-3 py-1.5 text-[11px] text-amber-200">
              <span className="inline-flex items-center gap-1.5">
                <Globe className="h-3 w-3 text-amber-300 shrink-0" />
                <span>
                  <code className="font-mono">:online</code> web search is currently OFF.
                </span>
              </span>
              <button
                type="button"
                onClick={onEnableOnlineSearch}
                className="rounded bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-400/40 px-2 py-0.5 font-medium text-emerald-200 cursor-pointer"
              >
                Enable :online search
              </button>
            </div>
          )}
          <div className="flex items-end gap-2">
            <textarea
              rows={2}
              value={localInput}
              onChange={(e) => setLocalInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleLocalSubmit(e);
                }
              }}
              placeholder="Send a prompt through QuixiGate… (Enter to send, Shift+Enter for newline)"
              className="flex-1 resize-none rounded-xl bg-slate-900 border border-white/10 px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-400"
            />
            {isStreaming ? (
              <button
                type="button"
                onClick={() => void stop()}
                className="rounded-xl bg-rose-500 hover:bg-rose-400 px-4 py-3 text-xs font-medium text-white inline-flex items-center gap-1.5 cursor-pointer"
              >
                <Square className="h-3.5 w-3.5" />
                Stop
              </button>
            ) : (
              <button
                type="submit"
                className="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 text-xs font-medium text-white inline-flex items-center gap-1.5 cursor-pointer"
              >
                <Send className="h-3.5 w-3.5" />
                Send
              </button>
            )}
          </div>
        </form>
      )}
    </div>
  );
}

export function PlaygroundArena({
  initialModel,
  onNavigateProviders,
}: {
  initialModel?: string | null;
  onNavigateProviders?: () => void;
}) {
  const [arenaMode, setArenaMode] = useState(false);
  const [modelA, setModelA] = useState(initialModel || '');
  const [modelB, setModelB] = useState('');
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [openRouterConnected, setOpenRouterConnected] = useState(false);
  const [routingModifier, setRoutingModifier] = useState<'none' | 'nitro' | 'floor'>('none');
  const [showControls, setShowControls] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [reasoningEffort, setReasoningEffort] = useState('default');
  const [onlineSearch, setOnlineSearch] = useState(true);
  const [arenaPrompt, setArenaPrompt] = useState('');
  const [broadcastSignal, setBroadcastSignal] = useState<{
    seq: number;
    text: string;
  } | null>(null);

  useEffect(() => {
    if (initialModel) setModelA(initialModel);
  }, [initialModel]);

  useEffect(() => {
    void api.getModelsBundle().then((b) => {
      const enabledProviders = new Set<string>(
        b.credentials
          .filter((c: CredentialRecord) => c.enabled)
          .map((c: CredentialRecord) => c.provider_id),
      );
      const hasOpenRouter = enabledProviders.has('openrouter');
      setOpenRouterConnected(hasOpenRouter);
      const routeIds = b.routes.map((r: ModelRoute) => r.alias);
      const localFirst = b.models
        .filter((m: CatalogModel) => m.is_local)
        .map((m: CatalogModel) => m.canonical_id);
      const presetIds = hasOpenRouter
        ? b.models
            .filter(
              (m: CatalogModel) =>
                Boolean(m.is_preset) && isModelRoutable(m, enabledProviders),
            )
            .map((m: CatalogModel) => m.canonical_id)
        : [];
      const configuredCloudIds = b.models
        .filter(
          (m: CatalogModel) =>
            !m.is_local && !m.is_preset && isModelRoutable(m, enabledProviders),
        )
        .map((m: CatalogModel) => m.canonical_id);
      const all = Array.from(
        new Set([...routeIds, ...localFirst, ...presetIds, ...configuredCloudIds]),
      );
      setModelOptions(all);
      if (!initialModel) {
        setModelA(all[0] ?? '');
      }
      setModelB(all[1] ?? all[0] ?? '');
    });
  }, [initialModel]);

  function handleArenaSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = arenaPrompt.trim();
    if (!trimmed) return;
    setArenaPrompt('');
    setBroadcastSignal((prev) => ({ seq: (prev?.seq ?? 0) + 1, text: trimmed }));
  }

  return (
    <div className="p-6 lg:p-8 flex flex-col h-[calc(100vh-1rem)] max-w-[1520px] mx-auto gap-4">
      {/* Header & Mode Switch */}
      <div className="flex flex-wrap items-center justify-between gap-3 shrink-0">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            {arenaMode ? 'Multi-Model Arena' : 'Interactive Playground'}
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Live streaming via <code className="text-indigo-300">POST /api/chat</code> with{' '}
            <code className="text-indigo-300">:online</code> web search, reasoning traces, and side-by-side seat comparison.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {openRouterConnected && (
            <select
              value={routingModifier}
              onChange={(e) =>
                setRoutingModifier(e.target.value as 'none' | 'nitro' | 'floor')
              }
              title="OpenRouter provider routing modifier"
              className="rounded-lg bg-[#0d111a] border border-violet-500/30 px-2.5 py-1.5 text-xs font-mono text-violet-200 focus:outline-none"
            >
              <option value="none">Route: Default</option>
              <option value="nitro">:nitro (Throughput)</option>
              <option value="floor">:floor (Lowest Cost)</option>
            </select>
          )}
          <button
            type="button"
            aria-pressed={onlineSearch}
            onClick={() => setOnlineSearch((v) => !v)}
            title={
              onlineSearch
                ? 'Server-side :online web search & URL fetch tools are active — click to disable'
                : 'Web search is currently OFF — click to enable :online web search'
            }
            className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition cursor-pointer ${
              onlineSearch
                ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-200 shadow-[0_0_12px_rgba(16,185,129,0.12)]'
                : 'bg-amber-500/10 hover:bg-amber-500/15 border-amber-500/35 text-amber-200'
            }`}
          >
            <Globe
              className={`h-3.5 w-3.5 ${
                onlineSearch ? 'text-emerald-400' : 'text-amber-300'
              }`}
            />
            <span>:online Search</span>
            <span
              className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold ${
                onlineSearch
                  ? 'bg-emerald-400/20 text-emerald-300'
                  : 'bg-amber-400/20 text-amber-200'
              }`}
            >
              {onlineSearch ? 'ON' : 'OFF'}
            </span>
          </button>

          <button
            type="button"
            onClick={() => setShowControls((v) => !v)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#0d111a] border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-200 cursor-pointer"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            Parameters
          </button>

          <button
            type="button"
            onClick={() => setArenaMode((v) => !v)}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-medium cursor-pointer ${
              arenaMode
                ? 'bg-indigo-500 text-white'
                : 'bg-[#0d111a] border border-white/10 text-slate-200 hover:text-white'
            }`}
          >
            <Columns2 className="h-3.5 w-3.5" />
            {arenaMode ? 'Arena Mode: Side-by-Side' : 'Switch to Arena'}
          </button>
        </div>
      </div>

      {/* Collapsible Parameter Controls */}
      {showControls && (
        <div className="rounded-xl bg-[#0d111a] border border-white/10 p-4 grid grid-cols-1 md:grid-cols-3 gap-4 shrink-0">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              System Prompt
            </label>
            <input
              type="text"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="You are a concise staff systems engineer..."
              className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Temperature ({temperature.toFixed(2)})
            </label>
            <input
              type="range"
              min="0"
              max="1.5"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full accent-indigo-500 mt-2"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Reasoning Effort
            </label>
            <select
              value={reasoningEffort}
              onChange={(e) => setReasoningEffort(e.target.value)}
              className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-xs text-white"
            >
              <option value="default">Model Default</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>
      )}

      {/* Single Pane or Dual Arena Grid */}
      <div
        className={`flex-1 min-h-0 grid gap-4 ${
          arenaMode ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'
        }`}
      >
        <ChatPane
          paneLabel={arenaMode ? 'Model A' : undefined}
          selectedModel={modelA}
          onSelectModel={setModelA}
          modelOptions={modelOptions}
          systemPrompt={systemPrompt}
          temperature={temperature}
          reasoningEffort={reasoningEffort}
          onlineSearch={onlineSearch}
          onEnableOnlineSearch={() => setOnlineSearch(true)}
          routingModifier={openRouterConnected ? routingModifier : 'none'}
          broadcastSignal={arenaMode ? broadcastSignal : null}
          hideComposer={arenaMode}
          onNavigateProviders={onNavigateProviders}
        />

        {arenaMode && (
          <ChatPane
            paneLabel="Model B"
            selectedModel={modelB}
            onSelectModel={setModelB}
            modelOptions={modelOptions}
            systemPrompt={systemPrompt}
            temperature={temperature}
            reasoningEffort={reasoningEffort}
            onlineSearch={onlineSearch}
            onEnableOnlineSearch={() => setOnlineSearch(true)}
            routingModifier={openRouterConnected ? routingModifier : 'none'}
            broadcastSignal={broadcastSignal}
            hideComposer
            onNavigateProviders={onNavigateProviders}
          />
        )}
      </div>

      {/* Shared Arena Broadcast Composer */}
      {arenaMode && (
        <form
          onSubmit={handleArenaSubmit}
          className="shrink-0 rounded-2xl bg-[#0d111a] border border-indigo-500/30 p-3 flex items-center gap-3"
        >
          <input
            type="text"
            value={arenaPrompt}
            onChange={(e) => setArenaPrompt(e.target.value)}
            placeholder="Broadcast prompt simultaneously to Model A & Model B…"
            className="flex-1 rounded-xl bg-slate-950 border border-white/10 px-4 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-400"
          />
          <button
            type="submit"
            className="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-5 py-2.5 text-xs font-semibold text-white inline-flex items-center gap-2 cursor-pointer"
          >
            <Send className="h-3.5 w-3.5" />
            Stream Both
          </button>
        </form>
      )}
    </div>
  );
}
