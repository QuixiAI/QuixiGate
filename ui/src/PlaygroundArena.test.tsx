import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { UIMessage } from '@ai-sdk/react';
import type { ChatMessageMetadata } from './types';

type MockChatState = {
  messages: UIMessage<ChatMessageMetadata>[];
  status: 'ready' | 'submitted' | 'streaming' | 'error';
  sendMessage: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
  setMessages: ReturnType<typeof vi.fn>;
  error: Error | undefined;
  clearError: ReturnType<typeof vi.fn>;
};

let mockChatState: MockChatState;

vi.mock('@ai-sdk/react', () => ({
  useChat: () => mockChatState,
}));

import { PlaygroundArena } from './screens/PlaygroundArena';

describe('PlaygroundArena reasoning collapse inheritance & smart auto-scroll', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    window.localStorage.clear();
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          models: [
            {
              canonical_id: 'ollama/lfm2.5:24b',
              bare_id: 'lfm2.5:24b',
              author: 'ollama',
              display_name: 'lfm2.5:24b',
              providers: { ollama: 'lfm2.5:24b' },
              context_window: 32768,
              max_output_tokens: 4096,
              input_nanos_per_m: 0,
              output_nanos_per_m: 0,
              cache_read_nanos_per_m: 0,
              supports_tools: true,
              supports_reasoning: true,
              supports_vision: false,
              supports_structured_output: true,
              is_local: true,
            },
          ],
          routes: [],
          credentials: [],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    mockChatState = {
      messages: [],
      status: 'ready',
      sendMessage: vi.fn(),
      stop: vi.fn(),
      setMessages: vi.fn(),
      error: undefined,
      clearError: vi.fn(),
    };
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('defaults the next reasoning trace to collapsed when the current reasoning trace is collapsed', async () => {
    mockChatState.messages = [
      {
        id: 'u1',
        role: 'user',
        parts: [{ type: 'text', text: 'hello' }],
      },
      {
        id: 'a1',
        role: 'assistant',
        parts: [
          { type: 'reasoning', text: 'First thought trace body' },
          { type: 'text', text: 'First answer' },
        ],
      },
    ];

    const { rerender } = render(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);

    // First thought trace starts open by default
    expect(screen.getByText('First thought trace body')).toBeInTheDocument();
    const firstToggle = screen.getByRole('button', { name: /Reasoning Trace/i });
    expect(firstToggle).toHaveAttribute('aria-expanded', 'true');

    // User collapses the current thought trace
    fireEvent.click(firstToggle);
    expect(firstToggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByText('First thought trace body')).not.toBeInTheDocument();

    // Next assistant turn streams in with a new thought trace -> must start collapsed!
    mockChatState.status = 'streaming';
    mockChatState.messages = [
      ...mockChatState.messages,
      {
        id: 'u2',
        role: 'user',
        parts: [{ type: 'text', text: 'ok why not' }],
      },
      {
        id: 'a2',
        role: 'assistant',
        parts: [
          { type: 'reasoning', text: 'Second thought trace body' },
          { type: 'text', text: 'Second answer' },
        ],
      },
    ];
    rerender(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);

    const togglesAfterTurn2 = screen.getAllByRole('button', {
      name: /Reasoning Trace/i,
    });
    expect(togglesAfterTurn2).toHaveLength(2);
    expect(togglesAfterTurn2[1]).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByText('Second thought trace body')).not.toBeInTheDocument();

    // Now user expands the current (second) thought trace -> third turn should start expanded,
    // while the first thought trace remains collapsed.
    fireEvent.click(togglesAfterTurn2[1]);
    expect(screen.getByText('Second thought trace body')).toBeInTheDocument();

    mockChatState.messages = [
      ...mockChatState.messages,
      {
        id: 'u3',
        role: 'user',
        parts: [{ type: 'text', text: 'and next?' }],
      },
      {
        id: 'a3',
        role: 'assistant',
        parts: [{ type: 'reasoning', text: 'Third thought trace body' }],
      },
    ];
    rerender(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);

    const togglesAfterTurn3 = screen.getAllByRole('button', {
      name: /Reasoning Trace/i,
    });
    expect(togglesAfterTurn3).toHaveLength(3);
    expect(togglesAfterTurn3[0]).toHaveAttribute('aria-expanded', 'false');
    expect(togglesAfterTurn3[1]).toHaveAttribute('aria-expanded', 'true');
    expect(togglesAfterTurn3[2]).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('Third thought trace body')).toBeInTheDocument();
  });

  it('auto-scrolls while streaming, pauses when user scrolls up, and resumes when scrolled back to bottom', async () => {
    const { rerender } = render(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);
    const feed = screen.getByTestId('chat-messages-feed');

    let scrollHeightVal = 400;
    const clientHeightVal = 400;
    let scrollTopVal = 0;

    Object.defineProperty(feed, 'scrollHeight', {
      configurable: true,
      get: () => scrollHeightVal,
    });
    Object.defineProperty(feed, 'clientHeight', {
      configurable: true,
      get: () => clientHeightVal,
    });
    Object.defineProperty(feed, 'scrollTop', {
      configurable: true,
      get: () => scrollTopVal,
      set: (v: number) => {
        scrollTopVal = v;
      },
    });

    // Submit a prompt
    const input = screen.getByPlaceholderText(/Send a prompt through QuixiGate/i);
    fireEvent.change(input, { target: { value: 'stream something long' } });
    fireEvent.submit(input.closest('form')!);
    await waitFor(() => {
      expect(mockChatState.sendMessage).toHaveBeenCalledTimes(1);
    });

    // Chunk 1 streams in -> auto-scrolls to bottom (900)
    scrollHeightVal = 900;
    mockChatState.status = 'streaming';
    mockChatState.messages = [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'stream something long' }] },
      { id: 'a1', role: 'assistant', parts: [{ type: 'text', text: 'chunk 1' }] },
    ];
    rerender(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);
    expect(scrollTopVal).toBe(900);

    // Chunk 2 streams in -> auto-scrolls to bottom (1200)
    scrollHeightVal = 1200;
    mockChatState.messages = [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'stream something long' }] },
      { id: 'a1', role: 'assistant', parts: [{ type: 'text', text: 'chunk 1 + chunk 2' }] },
    ];
    rerender(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);
    expect(scrollTopVal).toBe(1200);

    // User scrolls UP mid-stream -> auto-scrolling must stop
    scrollTopVal = 500;
    fireEvent.scroll(feed);

    scrollHeightVal = 1600;
    mockChatState.messages = [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'stream something long' }] },
      {
        id: 'a1',
        role: 'assistant',
        parts: [{ type: 'text', text: 'chunk 1 + chunk 2 + chunk 3 while scrolled up' }],
      },
    ];
    rerender(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);
    expect(scrollTopVal).toBe(500);

    // User scrolls back to the bottom (within threshold: 1600 - 1190 - 400 = 10px <= 24px)
    scrollTopVal = 1190;
    fireEvent.scroll(feed);

    // Chunk 4 streams in -> auto-scrolling must resume!
    scrollHeightVal = 2000;
    mockChatState.messages = [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'stream something long' }] },
      {
        id: 'a1',
        role: 'assistant',
        parts: [{ type: 'text', text: 'chunk 1 + 2 + 3 + 4 resumed!' }],
      },
    ];
    rerender(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);
    expect(scrollTopVal).toBe(2000);
  });

  it('renders Markdown formatting and fenced code blocks with one-click Copy', async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: { writeText: writeTextMock },
    });

    mockChatState.messages = [
      {
        id: 'u-md',
        role: 'user',
        parts: [{ type: 'text', text: 'implement a* in rust' }],
      },
      {
        id: 'a-md',
        role: 'assistant',
        parts: [
          {
            type: 'text',
            text: [
              '1. **I don\'t have the ability to search the web.** Plain explanation.',
              '',
              '2. Perhaps you meant (*Barack Obama*?) with inline `PriorityQueue`.',
              '',
              '```rust',
              'fn a_star(start: Node) -> Vec<Node> {',
              '    vec![start]',
              '}',
              '```',
            ].join('\n'),
          },
        ],
      },
    ];

    render(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);

    // Bold & italic rendered as semantic elements without raw asterisks
    const strongEl = screen.getByText("I don't have the ability to search the web.");
    expect(strongEl.tagName).toBe('STRONG');
    const emEl = screen.getByText('Barack Obama');
    expect(emEl.tagName).toBe('EM');
    const inlineCodeEl = screen.getByText('PriorityQueue');
    expect(inlineCodeEl.tagName).toBe('CODE');

    // Fenced code block header + Copy button
    expect(screen.getByText('rust')).toBeInTheDocument();
    expect(screen.getByText(/fn a_star/)).toBeInTheDocument();

    const copyBtn = screen.getByRole('button', { name: /Copy code/i });
    fireEvent.click(copyBtn);
    await waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledWith(
        'fn a_star(start: Node) -> Vec<Node> {\n    vec![start]\n}',
      );
      expect(screen.getByText('Copied')).toBeInTheDocument();
    });
  });

  it('surfaces a one-click Enable :online search & retry action when :online is OFF', async () => {
    mockChatState.messages = [
      {
        id: 'u-web',
        role: 'user',
        parts: [{ type: 'text', text: 'search the web for articles about barack olama' }],
      },
      {
        id: 'a-web',
        role: 'assistant',
        parts: [
          {
            type: 'text',
            text: "1. **I don't have the ability to search the web.** In this conversation environment, I can't look up articles online.",
          },
        ],
      },
    ];

    render(<PlaygroundArena initialModel="ollama/lfm2.5:24b" />);

    // Toggle :online Search OFF
    const onlineToggle = screen.getByRole('button', { name: /:online Search/i });
    expect(onlineToggle).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(onlineToggle);
    expect(onlineToggle).toHaveAttribute('aria-pressed', 'false');

    // Contextual retry callout appears immediately
    const retryBtn = screen.getByRole('button', {
      name: /Enable :online search & retry/i,
    });
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(onlineToggle).toHaveAttribute('aria-pressed', 'true');
      expect(mockChatState.sendMessage).toHaveBeenCalledWith(
        { text: 'search the web for articles about barack olama' },
        expect.objectContaining({
          body: expect.objectContaining({
            model: 'ollama/lfm2.5:24b',
            online: true,
          }),
        }),
      );
    });
  });
});
