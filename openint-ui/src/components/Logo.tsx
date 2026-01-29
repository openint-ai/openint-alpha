import { Link } from 'react-router-dom';

/** Modern logo mark: "o" + "I" in a rounded square (openInt). Exported for hero and other reuse. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden
    >
      <rect width="40" height="40" rx="10" className="fill-brand-500" />
      <circle cx="16" cy="20" r="5" fill="white" opacity="0.95" />
      <rect x="26" y="12" width="3" height="16" rx="1.5" fill="white" opacity="0.95" />
    </svg>
  );
}

export default function Logo() {
  return (
    <Link
      to="/"
      className="inline-flex items-center gap-3 no-underline group focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 rounded-lg"
      aria-label="openInt.ai – Home"
    >
      <LogoMark className="w-11 h-11 sm:w-12 sm:h-12 shrink-0 group-hover:opacity-90 transition-opacity" />
      <span className="flex items-baseline gap-0.5">
        <span className="font-bold text-xl sm:text-2xl text-gray-900 tracking-tight group-hover:text-brand-700 transition-colors">
          openInt
        </span>
        <span className="font-semibold text-xl sm:text-2xl text-brand-600 tracking-tight group-hover:text-brand-700 transition-colors">
          .ai
        </span>
      </span>
    </Link>
  );
}
