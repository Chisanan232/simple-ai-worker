import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Sidebar for the Dev section
 */
const sidebars: SidebarsConfig = {
  dev: [
    {
      type: 'doc',
      id: 'development',
      label: '🚀 Development',
    },
    {
      type: 'doc',
      id: 'requirements',
      label: '📋 Requirements',
    },
    {
      type: 'doc',
      id: 'workflow',
      label: '🔄 Development Workflow',
    },
    {
      type: 'doc',
      id: 'coding-style',
      label: '🎨 Coding Styles and Rules',
    },
    {
      type: 'doc',
      id: 'type-checking',
      label: '🔍 Type Checking with MyPy',
    },
    {
      type: 'category',
      label: '🏗️ Architecture',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'architecture',
          label: '🏛️ Architecture Overview',
        },
      ],
    },
    {
      type: 'category',
      label: '⚙️ CI/CD Workflows',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'ci-cd/index',
          label: '🎯 CI/CD Overview',
        },
        {
          type: 'doc',
          id: 'ci-cd/continuous-integration',
          label: '🔄 Continuous Integration',
        },
        {
          type: 'doc',
          id: 'ci-cd/release-system',
          label: '🚀 Release System',
        },
        {
          type: 'doc',
          id: 'ci-cd/documentation-deployment',
          label: '📚 Documentation Deployment',
        },
        {
          type: 'doc',
          id: 'ci-cd/reusable-workflows',
          label: '♻️ Reusable Workflows',
        },
        {
          type: 'doc',
          id: 'ci-cd/additional-ci-workflows',
          label: '🛠️ Additional CI Workflows',
        },
        {
          type: 'doc',
          id: 'ci-cd/type-checking-workflow',
          label: '🔍 Type Checking Workflow',
        },
        {
          type: 'doc',
          id: 'ci-cd/developer-guide',
          label: '👨‍💻 Developer Guide',
        },
      ],
    },
  ],
};

export default sidebars;
