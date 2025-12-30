import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DeepLearningLab",
  description: "Interactive visualizers for deep learning basics",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <header className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
          <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-8 py-4">
            <div className="flex items-baseline gap-3">
              <div className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">DeepLearningLab</div>
              <div className="text-xs text-zinc-500 dark:text-zinc-400">visualizers</div>
            </div>
            <nav className="flex items-center gap-2 text-sm">
              <Link
                href="/"
                className="rounded-full border border-zinc-200 bg-white px-4 py-2 text-zinc-900 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50 dark:hover:bg-zinc-900"
              >
                Linear Regression
              </Link>
              <Link
                href="/logistic"
                className="rounded-full border border-zinc-200 bg-white px-4 py-2 text-zinc-900 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50 dark:hover:bg-zinc-900"
              >
                Logistic Regression
              </Link>
              <Link
                href="/cnn"
                className="rounded-full border border-zinc-200 bg-white px-4 py-2 text-zinc-900 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50 dark:hover:bg-zinc-900"
              >
                CNN / MNIST
              </Link>
            </nav>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
