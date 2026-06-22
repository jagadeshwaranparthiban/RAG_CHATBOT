import React, { useState, useRef, useEffect } from "react";
import { Send, Paperclip, X, FileText, Loader2, ChevronDown, AlertCircle } from "lucide-react";

// Point this at your FastAPI server.
const API_BASE = process.env.BACKEND_BASE_URL || "http://localhost:8000";

export default function RagChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [files, setFiles] = useState([]); // { name, chunks, status: 'uploading' | 'done' | 'error' }
  const [error, setError] = useState(null);

  const scrollRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isSending]);

  const totalChunks = files.reduce((sum, f) => sum + (f.chunks || 0), 0);
  const hasDocuments = files.some((f) => f.status === "done");

  async function handleFileUpload(fileList) {
    const selected = Array.from(fileList);
    if (selected.length === 0) return;

    setError(null);
    const pending = selected.map((f) => ({ name: f.name, chunks: 0, status: "uploading" }));
    setFiles((prev) => [...prev, ...pending]);

    const formData = new FormData();
    selected.forEach((f) => formData.append("files", f));

    try {
      const res = await fetch(`${API_BASE}/ingest`, { method: "POST", body: formData });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Upload failed (${res.status})`);
      }
      const data = await res.json();

      setFiles((prev) => {
        const next = [...prev];
        data.files.forEach((result) => {
          const idx = next.findIndex((f) => f.name === result.filename && f.status === "uploading");
          if (idx !== -1) {
            next[idx] = { name: result.filename, chunks: result.chunks_indexed, status: "done" };
          }
        });
        return next;
      });
    } catch (err) {
      setFiles((prev) =>
        prev.map((f) =>
          pending.some((p) => p.name === f.name) && f.status === "uploading"
            ? { ...f, status: "error" }
            : f
        )
      );
      setError(err.message || "Couldn't upload those files. Check that the backend is running.");
    }
  }

  function removeFile(name) {
    setFiles((prev) => prev.filter((f) => f.name !== name));
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || isSending) return;

    setError(null);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setIsSending(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5 }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources || [] },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: null, error: err.message || "Something went wrong." },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function autoResize(e) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  }

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 shrink-0">
        <div>
          <h1 className="text-sm font-semibold tracking-wide text-slate-100">Document Q&amp;A</h1>
          <p className="text-xs text-slate-500 font-mono mt-0.5">
            {hasDocuments ? `${files.filter((f) => f.status === "done").length} docs · ${totalChunks} chunks indexed` : "no documents indexed yet"}
          </p>
        </div>
        <span className="flex items-center gap-1.5 text-xs text-slate-500">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
          groq
        </span>
      </header>

      {/* Chat area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-2xl mx-auto flex flex-col gap-5">
          {messages.length === 0 && (
            <div className="text-center py-20">
              <p className="text-slate-500 text-sm">
                Upload a document below, then ask a question about it.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatBubble key={i} message={msg} />
          ))}

          {isSending && (
            <div className="flex items-center gap-2 text-slate-500 text-sm">
              <Loader2 size={14} className="animate-spin" />
              searching documents…
            </div>
          )}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="max-w-2xl mx-auto w-full px-6">
          <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-400/10 border border-amber-400/20 rounded-md px-3 py-2 mb-2">
            <AlertCircle size={13} className="shrink-0" />
            {error}
          </div>
        </div>
      )}

      {/* File chips */}
      {files.length > 0 && (
        <div className="max-w-2xl mx-auto w-full px-6">
          <div className="flex flex-wrap gap-2 mb-2">
            {files.map((f) => (
              <FileChip key={f.name} file={f} onRemove={() => removeFile(f.name)} />
            ))}
          </div>
        </div>
      )}

      {/* Input bar */}
      <div className="border-t border-slate-800 px-6 py-4 shrink-0">
        <div className="max-w-2xl mx-auto flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt"
            className="hidden"
            onChange={(e) => {
              handleFileUpload(e.target.files);
              e.target.value = "";
            }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="shrink-0 p-2.5 rounded-lg border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 transition-colors"
            aria-label="Attach documents"
          >
            <Paperclip size={16} />
          </button>

          <textarea
            ref={textareaRef}
            value={input}
            onChange={autoResize}
            onKeyDown={handleKeyDown}
            placeholder={hasDocuments ? "Ask a question about your documents…" : "Upload a document first…"}
            rows={1}
            className="flex-1 resize-none bg-slate-900 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-cyan-400/50 focus:ring-1 focus:ring-cyan-400/30 transition-colors max-h-40"
          />

          <button
            onClick={handleSend}
            disabled={!input.trim() || isSending}
            className="shrink-0 p-2.5 rounded-lg bg-cyan-400 text-slate-950 hover:bg-cyan-300 disabled:bg-slate-800 disabled:text-slate-600 transition-colors"
            aria-label="Send question"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}

function FileChip({ file, onRemove }) {
  const statusStyles = {
    uploading: "border-slate-700 text-slate-400",
    done: "border-slate-800 text-slate-400",
    error: "border-red-400/30 text-red-400",
  };

  return (
    <div className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md border bg-slate-900 ${statusStyles[file.status]}`}>
      {file.status === "uploading" ? (
        <Loader2 size={12} className="animate-spin shrink-0" />
      ) : (
        <FileText size={12} className="shrink-0" />
      )}
      <span className="max-w-[160px] truncate">{file.name}</span>
      {file.status === "done" && (
        <span className="font-mono text-slate-600">· {file.chunks}</span>
      )}
      {file.status === "error" && <span className="text-red-400">· failed</span>}
      <button onClick={onRemove} className="text-slate-600 hover:text-slate-300 ml-0.5" aria-label={`Remove ${file.name}`}>
        <X size={12} />
      </button>
    </div>
  );
}

function ChatBubble({ message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="bg-slate-800 text-slate-100 rounded-lg rounded-br-sm px-4 py-2.5 text-sm max-w-[80%]">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="flex justify-start">
        <div className="flex items-start gap-2 text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg rounded-bl-sm px-4 py-2.5 max-w-[80%]">
          <AlertCircle size={14} className="shrink-0 mt-0.5" />
          {message.error}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] flex flex-col gap-2">
        <div className="bg-slate-900 border border-slate-800 rounded-lg rounded-bl-sm px-4 py-2.5 text-sm text-slate-100 leading-relaxed whitespace-pre-wrap">
          {message.content}
        </div>
        {message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} />
        )}
      </div>
    </div>
  );
}

function SourceList({ sources }) {
  const [openIndex, setOpenIndex] = useState(null);

  return (
    <div className="flex flex-wrap gap-1.5">
      {sources.map((src, i) => (
        <div key={i} className="relative">
          <button
            onClick={() => setOpenIndex(openIndex === i ? null : i)}
            className="flex items-center gap-1 text-xs font-mono text-slate-500 hover:text-cyan-400 border border-slate-800 hover:border-cyan-400/30 rounded px-2 py-1 transition-colors"
          >
            <span className="text-cyan-400/70">[{i + 1}]</span>
            <span className="max-w-[100px] truncate">{src.source}</span>
            <ChevronDown size={11} className={`transition-transform ${openIndex === i ? "rotate-180" : ""}`} />
          </button>

          {openIndex === i && (
            <div className="absolute z-10 bottom-full mb-1.5 left-0 w-64 bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-lg shadow-black/40">
              <div className="flex items-center justify-between text-xs font-mono text-slate-500 mb-1.5">
                <span className="truncate">{src.source}</span>
                <span className="text-cyan-400 shrink-0 ml-2">{src.score}</span>
              </div>
              <p className="text-xs text-slate-400">
                chunk #{src.chunk_index} · relevance score shown is cosine similarity
              </p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}