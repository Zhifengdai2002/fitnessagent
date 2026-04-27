import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FitnessAgent",
  description: "AI fitness planning workspace"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

