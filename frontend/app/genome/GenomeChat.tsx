import { locusAtom } from "@/app/genome/atoms";
import { useSetAtom } from "jotai";
import { omit } from "lodash";
import { useState } from "react";
import { toast } from "react-hot-toast";

export default function GenomeChat() {
  const setLocus = useSetAtom(locusAtom);
  const [messages, setMessages] = useState<
    {
      id: string;
      role: string;
      content: { type: string; text: string }[];
      locus?: string;
    }[]
  >([]);
  const [nextUserMessage, setNextUserMessage] = useState<string>("");
  const [nextGenomeMessage, setNextGenomeMessage] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function onSubmitNewUserMessage(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (nextUserMessage.trim() === "") {
      return;
    }

    setMessages((prevMessages) => {
      const newMessage = {
        id: `id-${prevMessages.length + 1}`,
        content: [{ type: "text", text: nextUserMessage }],
        role: "user",
      };

      return [...prevMessages, newMessage];
    });
    setNextUserMessage("");
    await new Promise(requestAnimationFrame);

    setIsSubmitting(true);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${process.env.NEXT_PUBLIC_BACKEND_AUTH_TOKEN}`,
          },
          body: JSON.stringify({
            messages: [
              ...messages
                .map((message) => omit(message, ["locus", "id"]))
                .filter((message) => message.role !== "tool"),
              {
                role: "user",
                content: [{ type: "text", text: nextUserMessage }],
              },
            ],
          }),
        }
      );

      if (!response.ok || !response.body) {
        toast.error("Unable to send message. Please try again later.");
        setNextUserMessage(nextUserMessage);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        toast.error("Unable to read response. Please try again later.");
        setNextUserMessage(nextUserMessage);
        return;
      }

      setTimeout(() => {
        window.scrollTo({
          top: document.body.scrollHeight,
          behavior: "smooth",
        });
      }, 0);

      const decoder = new TextDecoder();
      let buffer = "";
      let locus = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        let lastSnapshot = "";

        for (const line of lines) {
          if (line.trim() === "") {
            continue;
          }

          try {
            const json = JSON.parse(line);
            console.debug(json);

            setIsSubmitting(false);
            await new Promise(requestAnimationFrame);

            if (json.type === "text") {
              setNextGenomeMessage(json.snapshot.trim());
              lastSnapshot = json.snapshot.trim();
              // Allow React to flush the state update
              await new Promise(requestAnimationFrame);
            }

            if (
              json?.type === "content_block_stop" &&
              json?.content_block?.name === "get_snp_base_pairs"
            ) {
              locus = `chr${json?.content_block?.input?.chromosome}:${json?.content_block?.input?.position}`;
            }

            if (
              json?.type === "content_block_stop" &&
              json?.content_block?.type === "tool_use"
            ) {
              setMessages((prevMessages) => {
                const newMessage = {
                  id: `id-${prevMessages.length + 1}`,
                  content: [
                    {
                      type: "text",
                      text: `Using tool \`${json?.content_block?.name}\`...`,
                    },
                  ],
                  role: "tool",
                };
                return [...prevMessages, newMessage];
              });
            }

            if (
              json.type === "content_block_stop" &&
              lastSnapshot.trim() !== ""
            ) {
              if (locus) {
                setLocus(locus);
              }
              setMessages((prevMessages) => {
                const newMessage = {
                  id: `id-${prevMessages.length + 1}`,
                  content: [{ type: "text", text: lastSnapshot.trim() }],
                  role: "assistant",
                  locus,
                };
                lastSnapshot = "";
                locus = "";
                return [...prevMessages, newMessage];
              });
              setNextGenomeMessage("");
              await new Promise(requestAnimationFrame);
              setTimeout(() => {
                window.scrollTo({
                  top: document.body.scrollHeight,
                  behavior: "smooth",
                });
              }, 0);
            }
          } catch (err) {
            console.error(err);
            toast.error("Unable to parse response.");
          }
        }
      }

      // Clear the input field
      setNextUserMessage("");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="bg-gradient-to-r from-cyan-500/30 to-teal-500/30 rounded-2xl backdrop-blur-xl p-8 flex flex-col justify-center shadow-lg">
      <div className="flex items-center space-x-3 mb-4">
        <div className="w-3 h-3 bg-cyan-400 rounded-full"></div>
        <p className="font-mono text-sm opacity-80">Genome Chat</p>
      </div>

      {messages.map((message, i) => {
        if (message.role === "tool") {
          return (
            <div
              key={`id-${message.id}` || i}
              className="rounded-lg p-2 mb-3 max-w-lg ml-2"
            >
              <p className="text-xs">
                {message.content.map((it) => it.text).join("")}
              </p>
            </div>
          );
        }

        return (
          <div
            key={`id-${message.id}` || i}
            className={`bg-white/10 rounded-lg p-3 mb-3 max-w-lg whitespace-pre-line ${
              message.role === "user" ? "ml-auto" : ""
            }`}
          >
            <p className="text-sm">
              {message.content.map((it) => it.text).join("")}
            </p>
            {message.locus && (
              <small
                className="text-cyan-400 opacity-80 cursor-pointer hover:opacity-50"
                onClick={() => {
                  if (message.locus) {
                    setLocus(message.locus);
                    setTimeout(() => {
                      window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: "smooth",
                      });
                    }, 0);
                  }
                }}
              >
                See relevant genetic data at {message.locus}
              </small>
            )}
          </div>
        );
      })}

      {nextGenomeMessage !== "" && (
        <div className="bg-white/10 rounded-lg p-3 mb-3 max-w-lg">
          <p className="text-sm whitespace-pre-line">{nextGenomeMessage}</p>
        </div>
      )}

      {isSubmitting && (
        <div className="bg-white/10 rounded-lg p-3 mb-3 max-w-lg flex items-center space-x-3">
          <div className="flex space-x-1">
            {[...Array(3)].map((_, i) => (
              <span
                key={i}
                className={`block w-1 h-1 rounded-full bg-white animate-[bounce_1.2s_ease-in-out_infinite]`}
                style={{ animationDelay: `${i * 0.2}s` }}
              />
            ))}
          </div>
        </div>
      )}

      <div className="relative flex items-center mt-4 bg-white/5 rounded-lg p-2 border-white/30 border-1">
        <form
          onSubmit={onSubmitNewUserMessage}
          className="flex items-center w-full"
        >
          <input
            className="px-4 w-full focus:outline-none"
            placeholder="Talk to your genome..."
            disabled={isSubmitting}
            value={nextUserMessage}
            onChange={(e) => setNextUserMessage(e.target.value)}
            autoFocus
          />
          <button
            type="submit"
            disabled={nextUserMessage.trim() === "" || isSubmitting}
            className="ml-2 w-8 h-8 rounded-lg bg-cyan-500 flex items-center justify-center cursor-pointer hover:bg-cyan-400 transition-colors duration-200 disabled:bg-gray-500/50 disabled:cursor-not-allowed"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}
