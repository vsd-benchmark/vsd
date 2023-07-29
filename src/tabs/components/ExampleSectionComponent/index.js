import { Typography, Table, Image} from "antd";

import _ from "lodash";
import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";
import './index.scss'

const allFiles = ((ctx) => {
    let keys = ctx.keys();
    let values = keys.map(ctx);
    return keys.reduce((o, k, i) => {
      o[k] = values[i];
      return o;
    }, {});
  })(require.context("../../../examples", true, /.*/));
  
const MODELS = ["clip", "beit", "argus", "dino"];

const responsive = {
  desktop: {
    breakpoint: { max: 3000, min: 1024 },
    items: 2,
    // slidesToSlide: 3 // optional, default to 1.
  },
  tablet: {
    breakpoint: { max: 1024, min: 464 },
    items: 2,
    // slidesToSlide: 2 // optional, default to 1.
  },
  mobile: {
    breakpoint: { max: 464, min: 0 },
    items: 1,
    // slidesToSlide: 1 // optional, default to 1.
  }
};

function getImagePath(set, path) {
    const key = `./${set}/${path.split("/").slice(-2).join("/")}`;
    const result = allFiles[key];
    return result;
  }

function ExampleItem({ set, itemData, sortAscending = true }) {
    let data = MODELS.map((curr, idx) => ({
      model: curr,
      rank: itemData[`rank_${curr}`],
      sim: itemData[`sim_${curr}`].toFixed(2),
      ...itemData,
      key: idx,
    }));
  
    data = _.sortBy(data, (x) => (sortAscending ? x["rank"] : -x["rank"]));
  
    const columns = [
      {
        title: "Model",
        dataIndex: "model",
        key: "model",
      },
      {
        title: "Rank",
        dataIndex: "rank",
        key: "rank",
      },
      {
        title: "Score",
        dataIndex: "sim",
        key: "sim",
      },
      {
        title: "Label",
        dataIndex: "value",
        key: "value",
      },
    ];
  
    return (
      <div className="example_item">
        <div className="example_images">
          <div className="example_image">
            Query
            <Image src={getImagePath(set, itemData["query"])} />
          </div>
          <div className="example_image">
            Candidate
            <Image src={getImagePath(set, itemData["gallery"])} />
          </div>
        </div>
        <Table
          dataSource={data}
          columns={columns}
          pagination={false}
          size="small"
          tableLayout="unset"
        />
      </div>
    );
  }
  

function ExamplesGroup({title, description, items, set}) {
  return (
      <div className="examples_group">
      <h5>{title}</h5>
      <p>{description}</p>
      <Carousel 
          className="example_items" 
          responsive={responsive} 
          showDots 
          infinite 
          minimumTouchDrag={20} 
          autoPlay={false} 
          autoPlaySpeed={6000}
        >
          {items.map((curr, idx) => (
            <ExampleItem set={set} itemData={curr} sortAscending={false} key={idx}/>
          ))}
      </Carousel>
    </div>
  )
}

export default function({set, title, description, truePoses, trueNegs, falseNegs, falsePoses}) {
    return (
        <section id={`examples_${set}`} className="container examples">
        <h4>{title}</h4>
        <div className="examples_description">
          {description}
        </div>
        <ExamplesGroup 
          title="Positives"
          description="Examples that were selected by the EDS Models to have high visual similarity and confirmed by experts."
          set={set}
          items={truePoses}
        />
        <ExamplesGroup 
          title="Negatives"
          description="Examples that were selected by the EDS Models to have high visual similarity but were annotated by experts to be negative."
          set={set}
          items={trueNegs}
        />
        <ExamplesGroup 
          title="Hard Positives"
          description="Examples annotated as positives, despite models not detecting high similarity."
          set={set}
          items={falseNegs}
        />
        <ExamplesGroup 
          title="Hard Negatives"
          description="Examples annotated as negatives, despite models detecting high similarity."
          set={set}
          items={falsePoses}
        />
      </section>
    )
}