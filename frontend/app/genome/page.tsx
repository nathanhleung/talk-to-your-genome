"use client";

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
  const [isLoading, setIsLoading] = useState(true);
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
    <main className="relative min-h-screen flex flex-col items-center justify-center text-white overflow-hidden">
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

      <div className="flex flex-col container mx-auto px-4 md:px-12 py-8 z-10 max-w-6xl flex-1">
        <div className="flex flex-1 flex-col md:flex-row items-center gap-2">
          <div className="flex flex-col gap-2">
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
          <div className="bg-white rounded-lg shadow-lg p-4 w-full mx-auto text-black text-center">
            <IGV />
            <p className="text-xs opacity-80 mt-1">
              Interactive Genome Visualization
            </p>
          </div>
        </div>

        <div className="relative w-full h-64 md:h-96 mt-8">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/30 to-teal-500/30 rounded-2xl backdrop-blur-xl p-8 pt-0 flex flex-col justify-center shadow-lg">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-3 h-3 bg-cyan-400 rounded-full"></div>
              <p className="font-mono text-sm opacity-80">Genome Chat</p>
            </div>

            <div className="bg-white/10 rounded-lg p-3 mb-3 max-w-xs">
              <p className="text-sm">
                What medications should I avoid with my genetics?
              </p>
            </div>

            <div className="bg-white/20 rounded-lg p-3 mb-3 ml-auto max-w-xs">
              <p className="text-sm">
                Based on your CYP2D6 variants, consider alternatives to codeine
                and tramadol.
              </p>
            </div>

            <div className="bg-white/10 rounded-lg p-3 max-w-xs">
              <p className="text-sm">What about my cardiovascular risk?</p>
            </div>

            <div className="flex items-center mt-4">
              <input
                className="flex-1 h-8 rounded-lg bg-white/5 flex items-center px-4"
                placeholder="Ask your genome..."
              />
              <div className="ml-2 w-8 h-8 rounded-lg bg-cyan-500 flex items-center justify-center cursor-pointer hover:bg-cyan-400 transition-colors duration-200">
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
              </div>
            </div>
          </div>
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
