const config = {
  title: "Skulk Test Harness",
  tagline: "End-to-end testing for Skulk clusters.",
  favicon: "img/skulk-logo.svg",

  url: "https://foxlight-foundation.github.io",
  baseUrl: process.env.DOCUSAURUS_BASE_URL || "/skulk-test-harness/",
  organizationName: "Foxlight-Foundation",
  projectName: "skulk-test-harness",

  onBrokenLinks: "throw",
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js"),
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],

  themes: ["@docusaurus/theme-mermaid"],

  themeConfig: {
    navbar: {
      title: "Harness",
      logo: {
        alt: "Skulk Logo",
        src: "img/skulk-logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Docs",
        },
        {
          href: "https://github.com/Foxlight-Foundation/skulk-test-harness",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Start",
          items: [
            { label: "Quickstart", to: "/quickstart" },
            { label: "What e2e testing means", to: "/concepts/e2e-testing" },
            { label: "First local run", to: "/guides/first-local-run" },
          ],
        },
        {
          title: "Reference",
          items: [
            { label: "Configuration", to: "/reference/configuration" },
            { label: "CLI", to: "/reference/cli" },
            { label: "Reports", to: "/reference/reports" },
          ],
        },
        {
          title: "Project",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/Foxlight-Foundation/skulk-test-harness",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Foxlight Foundation.`,
    },
    prism: {
      additionalLanguages: ["bash", "yaml", "json", "python"],
    },
  },
};

module.exports = config;
