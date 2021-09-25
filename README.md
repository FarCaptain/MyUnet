# MyUnet
A unet for ultrasonic weldingimage defect detection

This project uses a fully convolutional network U-net that performs well on small target image segmentation (especially medical image segmentation). In order to obtain better training results on a data set with extremely uneven data, we tried Combination of multiple parameters. This article uses the Dice coefficient as the value to measure the accuracy of the inspection task, and finally achieves a dice coefficient of 0.73on the task of detecting welding defects and marking the location of defects.

[![Unet](https://z3.ax1x.com/2021/09/25/4sSIDe.png)](https://imgtu.com/i/4sSIDe)

Actually implemented the model into a multi-user management system powered by PyQt.

[![4spKa9.png](https://z3.ax1x.com/2021/09/25/4spKa9.png)](https://imgtu.com/i/4spKa9)

[![4sp18x.png](https://z3.ax1x.com/2021/09/25/4sp18x.png)](https://imgtu.com/i/4sp18x)

[![4sperF.png](https://z3.ax1x.com/2021/09/25/4sperF.png)](https://imgtu.com/i/4sperF)

I would only share the core code of training the model though.