import React, {useContext, useMemo} from 'react';
import {Box, Text} from 'ink';
import SelectInput from 'ink-select-input';
import gradient from 'gradient-string';
import {AppContext} from '../app.mjs';
const h = React.createElement;

const palette = gradient(['#ff6a88', '#ff99ac', '#f2f5f7']);

const menuItems = [
  {
    label: '✨  Run AI commit workflow',
    value: 'workflow',
    hint: 'Stage, curate and craft commits with live telemetry.',
  },
  {
    label: '⚙️  Configure providers & models',
    value: 'configure',
    hint: 'Pick providers, endpoints and API keys with rich menus.',
  },
  {
    label: '🧪  Benchmark providers',
    value: 'benchmark',
    hint: 'Compare latency, quality and cost across your keys.',
  },
  {
    label: '🚪  Exit',
    value: 'exit',
    hint: 'Return to your terminal.',
  },
];

const MenuItem = ({isSelected, label, hint}) =>
  h(
    Box,
    {flexDirection: 'column'},
    h(Text, {color: isSelected ? 'cyanBright' : 'white'}, label),
    hint ? h(Text, {dimColor: true}, hint) : null,
  );

export default function MainMenu({onNavigate}) {
  const {bootstrap} = useContext(AppContext);
  const repoInfo = useMemo(() => {
    if (!bootstrap) {
      return null;
    }
    const provider = bootstrap?.config?.provider || 'openai';
    const model = bootstrap?.config?.model;
    return {
      repo: bootstrap.repoRoot,
      provider,
      model,
    };
  }, [bootstrap]);

  return h(
    Box,
    {flexDirection: 'column', padding: 1, gap: 1, borderStyle: 'round', borderColor: 'cyan'},
    h(
      Box,
      {flexDirection: 'column'},
      h(Text, null, palette.multiline('kcmt ✨')),
      h(Text, {dimColor: true}, 'UI: TUI (Ink)'),
      repoInfo
        ? h(
            Text,
            {dimColor: true},
            `Repo: ${repoInfo.repo} • Provider: ${repoInfo.provider} • Model: ${repoInfo.model}`,
          )
        : null,
    ),
    h(SelectInput, {
      onSelect: item => onNavigate(item.value),
      items: menuItems.map(item => ({label: item.label, value: item.value, hint: item.hint})),
      itemComponent: MenuItem,
    }),
  );
}
