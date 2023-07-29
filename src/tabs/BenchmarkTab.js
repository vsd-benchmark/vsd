import {
    Anchor,
    Typography,
  } from "antd";

  
import ExamplesSectionComponent from "./components/ExampleSectionComponent";

import BibTex from "./components/BibTexSection";
import catalogFalseNegs from "../examples/catalog/false_neg.json";
import catalogFalsePoses from "../examples/catalog/false_pos.json";
import catalogTruePoses from "../examples/catalog/true_pos.json";
import catalogTrueNegs from "../examples/catalog/true_neg.json";

import wildFalseNegs from "../examples/wild/false_neg.json";
import wildFalsePoses from "../examples/wild/false_pos.json";
import wildTruePoses from "../examples/wild/true_pos.json";
import wildTrueNegs from "../examples/wild/true_neg.json";

const {Paragraph, Link} = Typography;

const BENCHMARK_ANCHORS = [
  {
    key: "leaderboard",
    href: "#leaderboard",
    title: "Leaderboard",
  },
  {
    key: "Fashion-Catalog",
    href: "#examples_catalog",
    title: "Examples - Catalog",
  },
  {
    key: "Fashion-Consumer",
    href: "#examples_wild",
    title: "Examples - Wild",
  },
  {
    key: "bibtx",
    href: "#bibtx",
    title: "BibTeX",
  },
];

export default function BenchmarkTab({}) {
  return (
    <>
      <div className="anchors-container">
        <Anchor
          items={BENCHMARK_ANCHORS}
          className="anchors"
          direction="horizontal"
          offsetTop={35}
          style={{ zIndex: 100 }}
          rootClassName="anchors-container"
        />
      </div>
      <section id="intro" className="container">
        <div className="tasks">
          <h4>Tasks</h4>
          <Paragraph>
            This site displays the VSD datasets and the benchmark leaderboards on several tasks:
            <ul>
              <li>VSD Fashion Dataset
                <ol>
                  <li>In Catalog Retrieval
                    <ol>
                      <li>Zero Shot Retrieval Task</li>
                      <li>Open Catalog Training Retrieval Task - Same queries appear in train and test.</li>
                      <li>Closed Catalog Training Retrieval Task - Queries in train and test do not intersect.</li>
                    </ol>
                  </li>
                  <li>Consumer-Catalog (Wild) Retrieval
                  <ol>
                      <li>Zero Shot Retrieval Task</li>
                    </ol>
                  </li>
                </ol>
              </li>
            </ul>
          </Paragraph>
        </div>
        <h4>What is VSD?</h4>
        <Paragraph className="what_is_vsd">
        Visual similarity measures the perceptual agreement between two objects based on their visual appearance. Two objects can be similar or dissimilar based on their color, shape, size, pattern, utility, and more.<br/>
        In fact, all of these factors and many others play a role in determining the degree of visual similarity between two objects with varying importance.<br/>
        Therefore, defining perceived visual similarity based on these factors is challenging. Nonetheless, learning visual similarities is a key building block for many practical utilities such as search, recommendations, etc.<br/><br/>
        We differentiate the identification task and the discovery task. In the identification task; given a query image of an object, the identification task deals with retrieving images of an <strong>identical</strong> object taken under various conditions. However, in the discovery task, given an image of a certain object, VSD retrieves images of different objects with high perceptual visual similarity.
        </Paragraph>
      </section>
      <section id="leaderboard" className="leaderboard container">
        <h4>Leaderboard</h4>
        <Paragraph>
          Benchmark{" "}
          <Link href="https://huggingface.co/spaces/vsd-benchmark/vsd_leaderboard">
            leaderboard
          </Link>{" "}
          reported from HuggingFace on VSD-benchmark datasets.
          <br />
          Report metrics on your HuggingFace Model to include them in our table.{" "}
          <Link href="https://huggingface.co/docs/hub/model-cards#evaluation-results">
            Learn how. 
          </Link> 
        &nbsp; Check out our <Link href="https://huggingface.co/vsd-benchmark/vsd_example">example model.</Link>
        </Paragraph>
        <div className="leaderboard-container">
          <gradio-app
            src="https://vsd-benchmark-vsd-leaderboard.hf.space"
            theme_mode="light"
            eager
          ></gradio-app>
        </div>
      </section>
      <ExamplesSectionComponent
        title="VSD Fashion - In Catalog Retrieval"
        description="In the Closed-catalog discovery benchmark, we build upon the ICRB dataset, where the query images are taken from the catalog, and the task is to retrieve images associated with different objects that are similar to the item in the query image."
        set="catalog"
        falseNegs={catalogFalseNegs}
        trueNegs={catalogTrueNegs}
        falsePoses={catalogFalsePoses}
        truePoses={catalogTruePoses}
      />
      <ExamplesSectionComponent
        title="VSD Fashion - Consumer-Catalog (Wild) Retrieval"
        description="Discovery of perceptually similar objects where the query images were taken in the wild. In this setting, we are given a “wild” query image
        of a cloth item and the task is to retrieve images of perceptually similar items from the ICRB dataset. We adopt wild images from the Consumer-to-shop Clothes
        Retrieval Benchmark (CCRB) dataset and match them with
        candidates from the ICRB dataset."
        set="wild"
        falseNegs={wildFalseNegs}
        trueNegs={wildTrueNegs}
        falsePoses={wildFalsePoses}
        truePoses={wildTruePoses}
      />
      <BibTex />
    </>
  );
}
