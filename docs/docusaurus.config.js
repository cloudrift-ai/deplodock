// @ts-check

const {themes} = require('prism-react-renderer');
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Deplodock',
  tagline: 'Benchmark and deploy optimized LLM models on GPU servers',
  favicon: 'img/favicon.ico',

  url: 'https://deplodock.docs.cloudrift.ai',
  baseUrl: '/',

  organizationName: 'CloudRift',
  projectName: 'Deplodock',

  onBrokenLinks: 'throw',
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

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
          editUrl: 'https://github.com/CloudRift/deplodock-docs',
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
        {name: 'description', content: 'Deplodock documentation - Benchmark and deploy optimized LLM models on GPU servers with vLLM or SGLang.'},
        {name: 'keywords', content: 'Deplodock, LLM deployment, vLLM, SGLang, GPU benchmarking, model optimization, CloudRift'},
        {property: 'og:type', content: 'website'},
        {property: 'og:site_name', content: 'Deplodock Documentation'},
        {property: 'og:description', content: 'Benchmark and deploy optimized LLM models on GPU servers with vLLM or SGLang.'},
        {name: 'twitter:card', content: 'summary_large_image'},
        {name: 'twitter:title', content: 'Deplodock Documentation'},
        {name: 'twitter:description', content: 'Benchmark and deploy optimized LLM models on GPU servers with vLLM or SGLang.'},
      ],
      navbar: {
        title: 'Deplodock',
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
