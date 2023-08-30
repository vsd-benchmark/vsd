const bibtex_string = `
@misc{barkan2023efficient,
  title={Efficient Discovery and Effective Evaluation of Visual Perceptual Similarity: A Benchmark and Beyond}, 
  author={Oren Barkan and Tal Reiss and Jonathan Weill and Ori Katz and Roy Hirsch and Itzik Malkiel and Noam Koenigstein},
  year={2023},
  eprint={2308.14753},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
`

export default function BibTex() {
    return (
      <section id="bibtx" className="container bibtex">
        <h4>BibTeX</h4>
        <code className="qoute_code">{bibtex_string}</code>
      </section>
    );
}