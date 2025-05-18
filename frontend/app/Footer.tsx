export default function Footer() {
  return (
    <footer className="p-8">
      <p className="text-center opacity-75 text-sm mt-8">
        &copy; 2025 Team 12 at{" "}
        <a
          href="https://www.outofpocket.health/ai-hackathon"
          target="_blank"
          rel="noopener noreferrer"
          className="text-cyan-400 hover:underline"
        >
          Out of Pocket&apos;s Healthcare AI Hackathon
        </a>
        <div className="text-xs mt-2">
          Powered by{" "}
          <a
            href="https://anthropic.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyan-400 hover:underline"
          >
            Anthropic
          </a>
          ,{" "}
          <a
            href="https://www.canvasmedical.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyan-400 hover:underline"
          >
            Canvas
          </a>
          ,{" "}
          <a
            href="https://igv.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyan-400 hover:underline"
          >
            IGV
          </a>
          ,{" "}
          <a
            href="https://www.snpedia.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyan-400 hover:underline"
          >
            SNPedia
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
          . View code on{" "}
          <a
            href="https://github.com/nathanhleung/talk-to-your-genome/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-cyan-400 hover:underline"
          >
            GitHub
          </a>
          .
        </div>
      </p>
    </footer>
  );
}
