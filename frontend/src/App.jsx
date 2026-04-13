import React, { useState, useRef, useEffect } from 'react';
import { Sun, Moon, AlertCircle } from 'lucide-react';

const API_URL = 'http://localhost:8000/chat';

function classNames(...classes) {
  return classes.filter(Boolean).join(' ');
}

function SourcePills({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {sources.map((src, i) => (
        <span key={i} className="px-2 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-100 rounded-full text-xs font-medium">
          [{src}]
        </span>
      ))}
    </div>
  );
}

function LoadingDots() {
  return (
    <span className="flex items-center gap-1">
      <span className="h-2 w-2 rounded-full bg-gray-500 dark:bg-gray-300 animate-pulse" />
      <span className="h-2 w-2 rounded-full bg-gray-500 dark:bg-gray-300 animate-pulse [animation-delay:150ms]" />
      <span className="h-2 w-2 rounded-full bg-gray-500 dark:bg-gray-300 animate-pulse [animation-delay:300ms]" />
    </span>
  );
}

function ChatBubble({ message, isUser, error, loading, sources }) {
  return (
    <div className={classNames(
      'flex',
      isUser ? 'justify-end' : 'justify-start',
      'mb-2'
    )}>
      <div className={classNames(
        'max-w-[70%] px-4 py-2 rounded-lg shadow',
        isUser
          ? 'bg-blue-500 text-white rounded-br-none'
          : error
          ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100 rounded-bl-none'
          : 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-bl-none'
      )}>
        {loading ? (
          <span className="flex items-center gap-2"><LoadingDots /> Loading...</span>
        ) : error ? (
          <span className="flex items-center gap-2"><AlertCircle className="w-4 h-4" /> {message}</span>
        ) : (
          <span>{message}</span>
        )}
        {!isUser && !loading && !error && <SourcePills sources={sources} />}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    { text: 'Hi! How can I help you today?', isUser: false }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [dark, setDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  const chatRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
  }, [dark]);

  useEffect(() => {
    chatRef.current?.scrollTo(0, chatRef.current.scrollHeight);
  }, [messages, loading]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    setMessages((msgs) => [...msgs, { text: input, isUser: true }]);
    setInput('');
    setLoading(true);
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input })
      });
      if (!res.ok) throw new Error('Server error');
      const data = await res.json();
      setMessages((msgs) => [
        ...msgs,
        { text: data.answer, isUser: false, sources: data.sources }
      ]);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { text: 'Sorry, something went wrong.', isUser: false, error: true }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
      <header className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
        <h1 className="text-lg font-bold">RAG Chatbot</h1>
        <button
          aria-label="Toggle dark mode"
          className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-800"
          onClick={() => setDark((d) => !d)}
        >
          {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>
      </header>
      <main className="flex-1 flex flex-col items-center justify-center px-2">
        <div ref={chatRef} className="w-full max-w-xl flex-1 overflow-y-auto py-6">
          {messages.map((msg, i) => (
            <ChatBubble
              key={i}
              message={msg.text}
              isUser={msg.isUser}
              error={msg.error}
              loading={i === messages.length - 1 && loading && !msg.isUser}
              sources={msg.sources}
            />
          ))}
          {loading && (
            <ChatBubble message="" isUser={false} loading />
          )}
        </div>
        <form onSubmit={sendMessage} className="w-full max-w-xl flex gap-2 py-4">
          <input
            className="flex-1 px-4 py-2 rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-400 dark:focus:ring-blue-600"
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
            autoFocus
          />
          <button
            type="submit"
            className="px-4 py-2 rounded bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:opacity-50"
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
