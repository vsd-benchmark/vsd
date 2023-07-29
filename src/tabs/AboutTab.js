import {
    Tag,
    Typography
  } from "antd";
import BibTex from "./components/BibTexSection";
import { LineChartOutlined, QuestionCircleOutlined, GithubOutlined, FileTextOutlined} from '@ant-design/icons';

const {Paragraph} = Typography;


export default function AboutTab({}) {
    return (
      <>
        <div className="container">
            <h4>
            Efficient Discovery and Effective Evaluation of Visual Perceptual Similarity: A Benchmark and Beyond - ICCV 2023
            </h4>
            <h5>Abstract</h5>
            <Paragraph className="abstract">
                Visual similarities discovery (VSD) is an important task with broad
                e-commerce applications. Given an image of a certain object, the
                goal of VSD is to retrieve images of different objects with high
                perceptual visual similarity. Although being a highly addressed
                problem, the evaluation of proposed methods for VSD is often based
                on a proxy of an identification-retrieval task, evaluating the
                ability of a model to retrieve different images of the same object.
                We posit that evaluating VSD methods based on identification tasks
                is limited, and faithful evaluation must rely on expert annotations.
                In this paper, we introduce the first large-scale fashion visual
                similarity benchmark dataset, consisting of more than 110K
                expert-annotated image pairs. Besides this major contribution, we
                share insight from the challenges we faced while curating this
                dataset. Based on these insights, we propose a novel and efficient
                labeling procedure that can be applied to any dataset. We analyze
                its limitations and inductive biases, and based on this analysis, we
                propose metrics mitigating these limitations.
            </Paragraph>
        </div>
        <BibTex/>
      </>
    )
}