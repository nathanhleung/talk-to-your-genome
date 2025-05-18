"use client";

import GenomeChat from "@/app/genome/GenomeChat";
import clsx from "clsx";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";

const IGV = dynamic(() => import("./IGV"), { ssr: false });

const LOADING_SUBTITLES = [
  "Reading base pairs...",
  "Translating DNA to protein...",
  "Analyzing genetic variants...",
  "Predicting drug interactions...",
  "Generating personalized insights...",
  "Preparing your genome report...",
  "Finalizing your genome analysis...",
  "Ready to talk to your genome!",
];

export default function Genome() {
  const [isLoading, setIsLoading] = useState(
    process.env.NODE_ENV !== "development"
  );
  const [loadingSubtitle, setLoadingSubtitle] = useState(LOADING_SUBTITLES[0]);
  useEffect(() => {
    const interval = setInterval(() => {
      setLoadingSubtitle((prev) => {
        const currentIndex = LOADING_SUBTITLES.indexOf(prev);
        const nextIndex =
          currentIndex === LOADING_SUBTITLES.length - 1
            ? currentIndex
            : currentIndex + 1;

        if (nextIndex === LOADING_SUBTITLES.length - 1) {
          setIsLoading(false);
          clearInterval(interval);
        }

        return LOADING_SUBTITLES[nextIndex];
      });
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <main className="relative min-h-screen flex flex-col items-center justify-center text-white py-4">
      <div>
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-900 via-purple-900 to-blue-900 z-0"></div>
        <div className="absolute inset-0 bg-[radial-gradient(#fff3_1px,transparent_1px)] [background-size:16px_16px]"></div>
      </div>

      <div
        className={clsx(
          `fixed inset-0 z-20 transition-all`,
          isLoading
            ? "opacity-95 backdrop-blur"
            : "pointer-events-none opacity-0"
        )}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-900 via-purple-900 to-blue-900"></div>
        <div className="absolute inset-0 bg-[radial-gradient(#fff3_1px,transparent_1px)] [background-size:16px_16px]"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex items-center justify-center flex-col">
            <svg
              className="animate-spin h-8 w-8 text-cyan-400"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                d="M4 12a8 8 0 1 1 16 0A8 8 0 0 1 4 12zm2.5-1h9a2.5 2.5 0 1 1-5 0h-4a2.5 2.5 0 0 1-4.5-1z"
                fill="currentColor"
              />
            </svg>
            <h4 className="text-white text-xl mt-4">Processing genome...</h4>
            <p className="text-white/80 text-sm mt-2">{loadingSubtitle}</p>
          </div>
        </div>
      </div>

      <div className="relative flex flex-col container mx-auto px-4 md:px-12 py-8 max-w-6xl flex-1">
        <div className="flex flex-1 flex-col gap-2">
          <Link className="text-sm opacity-75" href="/">
            ← Back to Home
          </Link>
          <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight mt-2">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-teal-300">
              Talk
            </span>{" "}
            to Your Genome
          </h1>
        </div>

        <div className="mt-4">
          <GenomeChat />
        </div>

        <div className="rounded-lg shadow-lg p-4 w-full mx-auto text-black text-center bg-white opacity-95 backdrop-blur-xl mt-8">
          <IGV />
          <p className="text-xs opacity-80 mt-1">
            Interactive Genome Visualization
          </p>
        </div>

        <div>
          <p className="text-center opacity-80 text-xs mt-8">
            &copy; 2025 Team 12. Powered by{" "}
            <a
              href="https://anthropic.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:underline"
            >
              Anthropic
            </a>{" "}
            and{" "}
            <a
              href="https://pharmcat.org/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:underline"
            >
              PharmCAT
            </a>
            .
          </p>
        </div>
      </div>
    </main>
  );
}
