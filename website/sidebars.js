module.exports = {
  tutorialSidebar: [
    "index",
    "quickstart",
    {
      type: "category",
      label: "Concepts",
      collapsed: false,
      items: [
        "concepts/skulk-basics",
        "concepts/e2e-testing",
        "concepts/harness-model",
      ],
    },
    {
      type: "category",
      label: "Guides",
      collapsed: false,
      items: [
        "guides/first-local-run",
        "guides/write-model-set",
        "guides/write-test-set",
        "guides/compare-runs",
        "guides/submit-to-the-ledger",
        "guides/fleet-coordination",
        "guides/fresh-install-qualification",
        "guides/run-foxlight-profile",
        "guides/stability-suites",
      ],
    },
    {
      type: "category",
      label: "Reference",
      collapsed: false,
      items: [
        "reference/configuration",
        "reference/cli",
        "reference/model-sets",
        "reference/test-sets",
        "reference/reports",
      ],
    },
    "troubleshooting",
  ],
};
