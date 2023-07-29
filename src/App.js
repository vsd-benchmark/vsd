import Github from "./github.png";
import Arxiv from "./arxiv.jpg";

import "./App.scss";

import {theme, Tabs, Tag} from 'antd';
import { LineChartOutlined, QuestionCircleOutlined, GithubOutlined, FileTextOutlined} from '@ant-design/icons';


import StickyBox from 'react-sticky-box';

import _ from "lodash";
import { useEffect, useRef, useState } from "react";
import AboutTab from "./tabs/AboutTab";
import BenchmarkTab from "./tabs/BenchmarkTab";


const OPENU = "The Open University";
const HEBREW_UNI = "The Hebrew Univeristy of Jerusalem";
const TA_UNI = "Tel Aviv University";
const TECHNION = "Technion";

const TABS = [
  {
    key: "benchmark",
    label: (<span><LineChartOutlined/>Benchmark</span>),
    children: (<BenchmarkTab/>),
  },
  {
    key: "about",
    label: (<span><QuestionCircleOutlined />About</span>),
    children: (<AboutTab/>),
  }
]


function ReferenceLink({ icon: Icon, text, link = "" }) {
  return (
      <a className="link-item github" href={link}>
        <Tag className="reference-tag" icon={<Icon className="reference-icon"/>}>
              {text}
        </Tag>
      </a>

  );
}

function App() {
  const {
    token: { colorBgContainer },
  } = theme.useToken();

  const renderTabBar = (props, DefaultTabBar) => (
    <StickyBox
      offsetTop={0}
      offsetBottom={20}
      style={{
        zIndex: 100,
      }}
    >
      <DefaultTabBar
        {...props}
        style={{
          background: colorBgContainer,
        }}
      />
    </StickyBox>
  );

  return (
    <>
      <section className="title-section container container-center">
        <h2>Visual Similiarity Discovery</h2>
        <h5>Datasets and benchmarks</h5>
        <div className="links">
            <ReferenceLink icon={GithubOutlined} text="Code" link="https://github.com/vsd-benchmark/vsd"/>
            <ReferenceLink icon={FileTextOutlined} text="Paper" />
        </div>
      </section>
      <Tabs 
          animated 
          centered 
          renderTabBar={renderTabBar} 
          items={TABS}>
      </Tabs>
    </>
  );
}

export default App;
