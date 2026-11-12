import { type ReactNode, useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface CodeBlockProps {
  language: string;
  code: string;
}

function FencedCodeBlock({ language, code }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code);
      } else {
        const el = document.createElement('textarea');
        el.value = code;
        el.style.position = 'fixed';
        el.style.opacity = '0';
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Ignore clipboard permission errors in restricted contexts
    }
  }

  return (
    <div className="rounded-xl bg-[#070a10] border border-white/10 overflow-hidden my-2">
      <div className="px-3 py-1.5 bg-slate-900/90 border-b border-white/8 flex items-center justify-between gap-2 text-[11px] font-mono text-slate-400">
        <span>{language || 'code'}</span>
        <button
          type="button"
          onClick={() => void handleCopy()}
          aria-label="Copy code"
          className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] text-slate-300 hover:text-white hover:bg-white/10 transition cursor-pointer"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 text-emerald-400" />
              <span className="text-emerald-300">Copied</span>
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3 overflow-x-auto text-xs font-mono leading-relaxed text-slate-100">
        <code>{code}</code>
      </pre>
    </div>
  );
}

type TopSegment =
  | { kind: 'code'; language: string; code: string }
  | { kind: 'prose'; text: string };

/**
 * Splits markdown into prose and fenced code blocks (` ```lang ... ``` ` or `~~~`),
 * gracefully handling unclosed trailing fences while tokens are actively streaming.
 */
function splitFencedCodeBlocks(source: string): TopSegment[] {
  const normalized = source.replace(/\r\n/g, '\n');
  const lines = normalized.split('\n');
  const segments: TopSegment[] = [];

  let inFence = false;
  let fenceMarker: '```' | '~~~' = '```';
  let fenceLang = '';
  let buffer: string[] = [];

  const flushProse = () => {
    if (buffer.length > 0) {
      segments.push({ kind: 'prose', text: buffer.join('\n') });
      buffer = [];
    }
  };

  for (const line of lines) {
    if (!inFence) {
      const openMatch = line.match(/^\s*(`{3,}|~{3,})\s*([^\s`~]*)\s*$/);
      if (openMatch) {
        flushProse();
        inFence = true;
        fenceMarker = openMatch[1].startsWith('~') ? '~~~' : '```';
        fenceLang = openMatch[2] ?? '';
        continue;
      }
      buffer.push(line);
    } else {
      const closeRe =
        fenceMarker === '~~~' ? /^\s*~{3,}\s*$/ : /^\s*`{3,}\s*$/;
      if (closeRe.test(line)) {
        segments.push({
          kind: 'code',
          language: fenceLang,
          code: buffer.join('\n'),
        });
        buffer = [];
        inFence = false;
        fenceLang = '';
      } else {
        buffer.push(line);
      }
    }
  }

  if (inFence) {
    // Streaming unclosed code block -> render immediately as a live code card
    segments.push({
      kind: 'code',
      language: fenceLang,
      code: buffer.join('\n'),
    });
  } else {
    flushProse();
  }

  return segments;
}

const INLINE_PATTERN =
  /`([^`\n]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)|\*\*\*(?!\s)([\s\S]+?)(?<!\s)\*\*\*|\*\*(?!\s)([\s\S]+?)(?<!\s)\*\*|(?<![A-Za-z0-9_])__(?!\s)([\s\S]+?)(?<!\s)__(?![A-Za-z0-9_])|~~(?!\s)([\s\S]+?)(?<!\s)~~|(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)|(?<![A-Za-z0-9_])_(?!\s)([^_\n]+?)(?<!\s)_(?![A-Za-z0-9_])|(https?:\/\/[^\s<>()]+)/g;

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let tokenIdx = 0;

  const regex = new RegExp(INLINE_PATTERN.source, 'g');
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const key = `${keyPrefix}-${tokenIdx++}`;

    if (match[1] !== undefined) {
      // Inline `code`
      nodes.push(
        <code
          key={key}
          className="rounded bg-white/10 border border-white/10 px-1.5 py-0.5 font-mono text-[12px] text-indigo-200"
        >
          {match[1]}
        </code>,
      );
    } else if (match[2] !== undefined && match[3] !== undefined) {
      // [label](https://...)
      nodes.push(
        <a
          key={key}
          href={match[3]}
          target="_blank"
          rel="noreferrer"
          className="text-indigo-300 underline underline-offset-2 hover:text-indigo-200"
        >
          {renderInline(match[2], key)}
        </a>,
      );
    } else if (match[4] !== undefined) {
      // ***bold italic***
      nodes.push(
        <strong key={key} className="font-semibold text-white italic">
          {renderInline(match[4], key)}
        </strong>,
      );
    } else if (match[5] !== undefined || match[6] !== undefined) {
      // **bold** or __bold__
      const inner = match[5] ?? match[6] ?? '';
      nodes.push(
        <strong key={key} className="font-semibold text-white">
          {renderInline(inner, key)}
        </strong>,
      );
    } else if (match[7] !== undefined) {
      // ~~strikethrough~~
      nodes.push(
        <del key={key} className="text-slate-400 line-through">
          {renderInline(match[7], key)}
        </del>,
      );
    } else if (match[8] !== undefined || match[9] !== undefined) {
      // *italic* or _italic_
      const inner = match[8] ?? match[9] ?? '';
      nodes.push(
        <em key={key} className="italic text-slate-100">
          {renderInline(inner, key)}
        </em>,
      );
    } else if (match[10] !== undefined) {
      // Bare URL — trim trailing sentence punctuation cleanly
      const rawUrl = match[10];
      const trimmedUrl = rawUrl.replace(/[.,;:!?]+$/, '');
      const trailing = rawUrl.slice(trimmedUrl.length);
      nodes.push(
        <a
          key={key}
          href={trimmedUrl}
          target="_blank"
          rel="noreferrer"
          className="text-indigo-300 underline underline-offset-2 hover:text-indigo-200 break-all"
        >
          {trimmedUrl}
        </a>,
      );
      if (trailing) {
        nodes.push(trailing);
      }
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function renderLinesWithBreaks(lines: string[], keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  lines.forEach((line, idx) => {
    if (idx > 0) {
      out.push(<br key={`${keyPrefix}-br-${idx}`} />);
    }
    out.push(...renderInline(line, `${keyPrefix}-l${idx}`));
  });
  return out;
}

function isTableSeparator(line: string): boolean {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
}

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

function renderProseBlocks(text: string, segKey: string): ReactNode[] {
  const lines = text.split('\n');
  const blocks: ReactNode[] = [];
  let i = 0;
  let blockIdx = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Skip blank lines
    if (line.trim() === '') {
      i++;
      continue;
    }

    // Horizontal rule
    if (/^\s*(?:-{3,}|_{3,}|\*{3,})\s*$/.test(line)) {
      blocks.push(
        <hr key={`${segKey}-hr-${blockIdx++}`} className="border-white/10 my-2.5" />,
      );
      i++;
      continue;
    }

    // Heading (# .. ######)
    const headingMatch = line.match(/^\s*(#{1,6})\s+(.+?)\s*$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const content = headingMatch[2];
      const key = `${segKey}-h-${blockIdx++}`;
      const cls =
        level === 1
          ? 'text-base font-semibold text-white mt-2 mb-1'
          : level === 2
            ? 'text-sm font-semibold text-white mt-2 mb-1'
            : 'text-xs font-semibold uppercase tracking-wide text-slate-200 mt-1.5 mb-0.5';
      blocks.push(
        <div key={key} role="heading" aria-level={level} className={cls}>
          {renderInline(content, key)}
        </div>,
      );
      i++;
      continue;
    }

    // GFM Table
    if (
      line.includes('|') &&
      i + 1 < lines.length &&
      isTableSeparator(lines[i + 1])
    ) {
      const headers = splitTableRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim() !== '') {
        rows.push(splitTableRow(lines[i]));
        i++;
      }
      const key = `${segKey}-tbl-${blockIdx++}`;
      blocks.push(
        <div
          key={key}
          className="overflow-x-auto rounded-xl border border-white/10 my-2"
        >
          <table className="min-w-full text-xs border-collapse">
            <thead className="bg-slate-900/90 text-slate-200">
              <tr>
                {headers.map((h, hIdx) => (
                  <th
                    key={`${key}-th-${hIdx}`}
                    className="px-3 py-1.5 text-left font-semibold border-b border-white/10"
                  >
                    {renderInline(h, `${key}-th-${hIdx}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/8">
              {rows.map((r, rIdx) => (
                <tr key={`${key}-tr-${rIdx}`}>
                  {headers.map((_, cIdx) => (
                    <td
                      key={`${key}-td-${rIdx}-${cIdx}`}
                      className="px-3 py-1.5 text-slate-200 align-top"
                    >
                      {renderInline(r[cIdx] ?? '', `${key}-td-${rIdx}-${cIdx}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const quoteLines: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^\s*>\s?/, ''));
        i++;
      }
      const key = `${segKey}-bq-${blockIdx++}`;
      blocks.push(
        <blockquote
          key={key}
          className="border-l-2 border-indigo-400/50 pl-3 py-0.5 text-slate-300 italic"
        >
          {renderLinesWithBreaks(quoteLines, key)}
        </blockquote>,
      );
      continue;
    }

    // Unordered list (- / * / +)
    const ulMatch = line.match(/^\s*[-*+]\s+(.+)$/);
    if (ulMatch) {
      const items: string[][] = [];
      while (i < lines.length) {
        const m = lines[i].match(/^\s*[-*+]\s+(.+)$/);
        if (m) {
          items.push([m[1]]);
          i++;
          while (
            i < lines.length &&
            lines[i].trim() !== '' &&
            /^\s{2,}\S/.test(lines[i]) &&
            !/^\s*[-*+]\s+/.test(lines[i]) &&
            !/^\s*\d+[.)]\s+/.test(lines[i])
          ) {
            items[items.length - 1].push(lines[i].trim());
            i++;
          }
          // Consume a single blank line if immediately followed by another unordered item
          if (
            i + 1 < lines.length &&
            lines[i].trim() === '' &&
            /^\s*[-*+]\s+/.test(lines[i + 1])
          ) {
            i++;
          }
        } else {
          break;
        }
      }
      const key = `${segKey}-ul-${blockIdx++}`;
      blocks.push(
        <ul key={key} className="list-disc pl-5 space-y-1.5 leading-relaxed">
          {items.map((itemLines, idx) => (
            <li key={`${key}-li-${idx}`}>
              {renderLinesWithBreaks(itemLines, `${key}-li-${idx}`)}
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    // Ordered list (1. / 1))
    const olMatch = line.match(/^\s*(\d+)[.)]\s+(.+)$/);
    if (olMatch) {
      const items: Array<{ num: number; lines: string[] }> = [];
      while (i < lines.length) {
        const m = lines[i].match(/^\s*(\d+)[.)]\s+(.+)$/);
        if (m) {
          items.push({ num: Number(m[1]), lines: [m[2]] });
          i++;
          while (
            i < lines.length &&
            lines[i].trim() !== '' &&
            /^\s{2,}\S/.test(lines[i]) &&
            !/^\s*[-*+]\s+/.test(lines[i]) &&
            !/^\s*\d+[.)]\s+/.test(lines[i])
          ) {
            items[items.length - 1].lines.push(lines[i].trim());
            i++;
          }
          // Consume a single blank line if immediately followed by another ordered item
          if (
            i + 1 < lines.length &&
            lines[i].trim() === '' &&
            /^\s*\d+[.)]\s+/.test(lines[i + 1])
          ) {
            i++;
          }
        } else {
          break;
        }
      }
      const key = `${segKey}-ol-${blockIdx++}`;
      const startNum = items[0]?.num ?? 1;
      blocks.push(
        <ol
          key={key}
          start={startNum}
          className="list-decimal pl-5 space-y-1.5 leading-relaxed"
        >
          {items.map((item, idx) => (
            <li key={`${key}-li-${idx}`} value={item.num}>
              {renderLinesWithBreaks(item.lines, `${key}-li-${idx}`)}
            </li>
          ))}
        </ol>,
      );
      continue;
    }

    // Regular paragraph
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !/^\s*(#{1,6})\s+/.test(lines[i]) &&
      !/^\s*(?:-{3,}|_{3,}|\*{3,})\s*$/.test(lines[i]) &&
      !/^\s*>\s?/.test(lines[i]) &&
      !/^\s*[-*+]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !(
        lines[i].includes('|') &&
        i + 1 < lines.length &&
        isTableSeparator(lines[i + 1])
      )
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    const key = `${segKey}-p-${blockIdx++}`;
    blocks.push(
      <p key={key} className="leading-relaxed">
        {renderLinesWithBreaks(paraLines, key)}
      </p>,
    );
  }

  return blocks;
}

export function MarkdownMessage({ text }: { text: string }) {
  const segments = splitFencedCodeBlocks(text);

  return (
    <div className="space-y-2.5 leading-relaxed break-words">
      {segments.map((seg, idx) =>
        seg.kind === 'code' ? (
          <FencedCodeBlock
            key={`seg-${idx}`}
            language={seg.language}
            code={seg.code}
          />
        ) : (
          renderProseBlocks(seg.text, `seg-${idx}`)
        ),
      )}
    </div>
  );
}
