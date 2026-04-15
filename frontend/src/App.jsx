import React, { useState, useRef, useEffect } from "react";
import {
  Sun,
  Moon,
  AlertCircle,
  Send,
  Bot,
  User,
  Sparkles,
  MessageSquare,
  FileText,
  Zap,
  BookOpen,
  Cpu,
} from "lucide-react";

const API_URL = "http://localhost:8000/chat";

/* ── Source Pills ── */
function SourcePills({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {sources.map((src, i) => (
        <span
          key={i}
          className="source-pill flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium"
        >
          <FileText className="w-3 h-3" />
          {src}
        </span>
      ))}
    </div>
  );
}

/* ── Loading Dots ── */
function LoadingDots() {
  return (
    <span className="flex items-center gap-1.5">
      <span className="loading-dot h-2 w-2 rounded-full bg-purple-400 dark:bg-purple-300" />
      <span className="loading-dot h-2 w-2 rounded-full bg-pink-400 dark:bg-pink-300" />
      <span className="loading-dot h-2 w-2 rounded-full bg-rose-400 dark:bg-rose-300" />
    </span>
  );
}

/* ── Answer Tabs (RAG + Direct) ── */
function AnswerTabs({ ragAnswer, directAnswer, sources }) {
  const [activeTab, setActiveTab] = useState("rag");
  const hasBoth = ragAnswer && directAnswer;

  return (
    <div>
      {/* Tab Buttons — only show when both answers exist */}
      {hasBoth && (
        <div className="flex gap-1.5 mb-2.5">
          <button
            onClick={() => setActiveTab("rag")}
            className={`tab-button ${activeTab === "rag" ? "tab-active" : ""}`}
          >
            <BookOpen className="w-3 h-3" />
            RAG Answer
          </button>
          <button
            onClick={() => setActiveTab("direct")}
            className={`tab-button ${activeTab === "direct" ? "tab-active" : ""}`}
          >
            <Cpu className="w-3 h-3" />
            Direct AI
          </button>
        </div>
      )}

      {/* Tab Content */}
      <span className="text-sm leading-relaxed whitespace-pre-wrap">
        {activeTab === "rag" || !hasBoth ? ragAnswer : directAnswer}
      </span>

      {/* Source pills only for RAG tab */}
      {(activeTab === "rag" || !hasBoth) && <SourcePills sources={sources} />}
    </div>
  );
}

/* ── Chat Bubble ── */
function ChatBubble({
  message,
  isUser,
  error,
  loading,
  sources,
  directAnswer,
}) {
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      {/* Avatar */}
      {!isUser && (
        <div className="flex-shrink-0 mr-2.5 mt-1">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-400 to-pink-400 dark:from-purple-500 dark:to-pink-500 flex items-center justify-center shadow-md">
            <Bot className="w-4 h-4 text-white" />
          </div>
        </div>
      )}

      <div
        className={[
          "max-w-[75%] px-4 py-3 rounded-2xl transition-all duration-200",
          isUser
            ? "bubble-user rounded-br-md"
            : error
              ? "bubble-error rounded-bl-md"
              : "bubble-bot rounded-bl-md",
        ].join(" ")}
      >
        {loading ? (
          <span className="flex items-center gap-2 py-0.5">
            <LoadingDots />
            <span className="text-sm text-gray-500 dark:text-gray-300">
              Thinking...
            </span>
          </span>
        ) : error ? (
          <span className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{message}</span>
          </span>
        ) : isUser ? (
          <span className="text-sm leading-relaxed whitespace-pre-wrap">
            {message}
          </span>
        ) : (
          <AnswerTabs
            ragAnswer={message}
            directAnswer={directAnswer}
            sources={sources}
          />
        )}
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="flex-shrink-0 ml-2.5 mt-1">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-400 to-cyan-400 dark:from-indigo-500 dark:to-cyan-500 flex items-center justify-center shadow-md">
            <User className="w-4 h-4 text-white" />
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Main App ── */
export default function App() {
  const [messages, setMessages] = useState([
    {
      text: "Hi! 👋 How can I help you today? Ask me anything about your documents.",
      isUser: false,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [dark, setDark] = useState(
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
  );
  const chatRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    chatRef.current?.scrollTo({
      top: chatRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, loading]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    setMessages((msgs) => [...msgs, { text: input, isUser: true }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      setMessages((msgs) => [
        ...msgs,
        {
          text: data.answer,
          isUser: false,
          sources: data.sources,
          directAnswer: data.direct_answer || null,
        },
      ]);
    } catch {
      setMessages((msgs) => [
        ...msgs,
        {
          text: "Sorry, something went wrong. Please try again.",
          isUser: false,
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className={`min-h-screen flex flex-col relative overflow-hidden transition-colors duration-500 ${
        dark
          ? "bg-gradient-animated-dark text-gray-100"
          : "bg-gradient-animated-light text-gray-900"
      }`}
    >
      {/* ── Decorative Blobs ── */}
      <div className="blob w-72 h-72 bg-purple-400 top-[-5rem] left-[-5rem]" />
      <div
        className="blob w-96 h-96 bg-pink-300 bottom-[-8rem] right-[-8rem]"
        style={{ animationDelay: "2s" }}
      />
      <div
        className="blob w-64 h-64 bg-sky-300 top-1/2 left-1/3"
        style={{ animationDelay: "4s" }}
      />

      {/* ── Header ── */}
      <header className="glass relative z-10 flex items-center justify-between px-5 py-3.5 rounded-none">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center shadow-lg">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold gradient-text leading-tight">
              RAG Chatbot
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
              <Zap className="w-3 h-3" />
              Powered by AI
            </p>
          </div>
        </div>
        <button
          id="theme-toggle"
          aria-label="Toggle dark mode"
          className="glass p-2.5 rounded-xl hover:scale-105 active:scale-95 transition-transform duration-200"
          onClick={() => setDark((d) => !d)}
        >
          {dark ? (
            <Sun className="w-5 h-5 text-amber-300" />
          ) : (
            <Moon className="w-5 h-5 text-indigo-500" />
          )}
        </button>
      </header>

      {/* ── Chat Area ── */}
      <main className="flex-1 flex flex-col items-center px-3 py-4 relative z-10">
        {/* Chat Container */}
        <div className="glass w-full max-w-2xl flex-1 flex flex-col rounded-2xl overflow-hidden">
          {/* Chat Header Bar */}
          <div className="px-5 py-3 border-b border-white/20 dark:border-purple-500/10 flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-purple-500 dark:text-purple-400" />
            <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
              Conversation
            </span>
            <span className="ml-auto text-xs text-gray-400 dark:text-gray-500">
              {messages.length} message{messages.length !== 1 ? "s" : ""}
            </span>
          </div>

          {/* Messages */}
          <div
            ref={chatRef}
            className="flex-1 overflow-y-auto px-5 py-4 custom-scrollbar"
          >
            {messages.map((msg, i) => (
              <ChatBubble
                key={i}
                message={msg.text}
                isUser={msg.isUser}
                error={msg.error}
                loading={false}
                sources={msg.sources}
                directAnswer={msg.directAnswer}
              />
            ))}
            {loading && <ChatBubble message="" isUser={false} loading />}
          </div>

          {/* Input Area */}
          <form
            onSubmit={sendMessage}
            className="px-4 py-3 border-t border-white/20 dark:border-purple-500/10 flex gap-2.5 items-center"
          >
            <input
              id="chat-input"
              className="glass-input flex-1 px-4 py-2.5 rounded-xl text-sm text-gray-800 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none"
              type="text"
              placeholder="Ask something about your documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              autoFocus
            />
            <button
              id="send-button"
              type="submit"
              className="btn-send p-2.5 rounded-xl font-semibold flex items-center justify-center"
              disabled={loading || !input.trim()}
              aria-label="Send message"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="relative z-10 text-center py-2.5 text-xs text-gray-400 dark:text-gray-500">
        Built with <span className="text-pink-400">♥</span> using RAG + Gemini
        AI
      </footer>
    </div>
  );
}
