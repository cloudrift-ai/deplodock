// @ts-check

const {themes} = require('prism-react-renderer');
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Emmy',
  tagline: 'Benchmark and deploy optimized LLM models on GPU servers',
  favicon: 'img/favicon.ico',

  url: 'https://emmy.docs.cloudrift.ai',
  baseUrl: '/',

  organizationName: 'CloudRift',
  projectName: 'Emmy',

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/CloudRift/emmy-docs',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/cloudrift-social-card.webp',
      metadata: [
        {name: 'description', content: 'Emmy documentation - Benchmark and deploy optimized LLM models on GPU servers with vLLM or SGLang.'},
        {name: 'keywords', content: 'Emmy, LLM deployment, vLLM, SGLang, GPU benchmarking, model optimization, CloudRift'},
        {property: 'og:type', content: 'website'},
        {property: 'og:site_name', content: 'Emmy Documentation'},
        {property: 'og:description', content: 'Benchmark and deploy optimized LLM models on GPU servers with vLLM or SGLang.'},
        {name: 'twitter:card', content: 'summary_large_image'},
        {name: 'twitter:title', content: 'Emmy Documentation'},
        {name: 'twitter:description', content: 'Benchmark and deploy optimized LLM models on GPU servers with vLLM or SGLang.'},
      ],
      navbar: {
        title: 'Emmy',
        logo: {
          alt: 'CloudRift Logo',
          src: 'img/cloudrift_vector.svg',
        },
        items: [
          {
            href: 'https://www.cloudrift.ai/',
            label: 'CloudRift.ai',
            position: 'left',
          },
          {
            href: 'https://docs.cloudrift.ai',
            label: 'CloudRift Docs',
            position: 'left',
          },
          {
            href: 'https://x.com/CloudRiftAI',
            label: 'Follow on X',
            position: 'right',
            className: 'header-x-link',
            'aria-label': 'Follow CloudRift on X',
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
