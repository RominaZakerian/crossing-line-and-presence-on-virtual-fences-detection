# crossing-line-and-presence-on-virtual-fences-detection
detecting crossing line and presence on virtual fences using object tracking models.

In this project, I implemented the detection of the crossing a virtual line and presence on the virtual fences using object detection and object tracking models.
The coordinates of the line and virtual fences were given in the policy.json file.
The result for detecting crossing line is shown as below:

https://github.com/user-attachments/assets/f5a36799-ce53-45a8-b43c-486729c52616

As you can see, this code, each person is detected with a bounding box and assigns a unique ID for each person in the video (tracking part) and after crossing the line, the line color is changed to red and remains until 3 seconds. 




The result for presence on virtual fences is shown as below:

https://github.com/user-attachments/assets/b92ccc49-3dda-487a-866a-6f676e62242e

Same as preveious, If a person is present on the virtual fence, the color of the fence will change to red and the number of people present on the fence will also be displayed.
And if there is not any person in the fence, the color will be green and the number will be zero.


If you have any question, please don't hesitate to ask me. 
Email: heliazakeriyan@gmail.com
And if this repository was helpful, please star it.
