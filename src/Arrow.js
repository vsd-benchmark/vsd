
import { CSSProperties } from 'react'

// const arrowSize = 40
// const arrowOffset = 40

// const arrow = {
//   color: '#fff',
//   fontSize: arrowSize,
//   height: arrowSize,
//   width: arrowSize,
//   zIndex: 1
// }

// const styles = Stylesheet({
//   leftArrow: {
//     ...arrow,
//     left: arrowOffset
//   },
//   rightArrow: {
//     ...arrow,
//     right: arrowOffset
//   }
// })

// Arrow.jsx
import { LeftOutlined, RightOutlined } from '@ant-design/icons'


// Antd is doing some interesting things here. Using LeftOutlined and RightOutlined
// directly without wrapping them in this component doesn't work. Additionally, if
// we don't add currentSlide and slideCount to the pops, we get console errors.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const Arrow = ({ currentSlide, direction, slideCount, ...carouselProps }) =>
  direction === 'left' ? (
    <LeftOutlined {...carouselProps}/>
  ) : (
    <RightOutlined {...carouselProps}/>
  )

export default Arrow;