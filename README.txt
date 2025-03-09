CAPTCHAs are one of the most common tools employed to prevent automated bots from abusing online services, from web scraping to large-scale DDoS attacks. This project demonstrates that modern deep learning techniques can effectively learn to solve certain types of CAPTCHAs—in particular, those that require users to click on specific regions of an image containing multiple drawn elements.

In this repository, you will find a proof-of-concept deep learning model that has been trained to locate the position of drawn elements (or “draws”) within a larger image, mimicking the behavior required by some CAPTCHA systems. The model learns to identify the correct location of the drawn elements, which are typically presented in a corner of the image, by correlating the drawn template with the field where multiple draws are present.

The motivation behind this project stems from a genuine curiosity about the robustness of CAPTCHA systems. While CAPTCHAs have long been considered a reliable way to distinguish humans from bots, our experiments indicate that even relatively simple deep learning models can compromise these systems with only a few hours of development effort. This raises important questions about the effectiveness of current CAPTCHA-based security and suggests that alternative methods for user verification may be necessary.

Please note, this repository serves as a demonstration of potential vulnerabilities in CAPTCHA systems rather than an operational tool. Its primary purpose is to illustrate that if such deep learning solutions are achievable with limited resources, then more robust and multi-layered security measures are essential to ensure the integrity of web services.


In the Dev folder you'll find all the Tools used to assure a good training of the deep Learning model which is in the Training.ipynb file.

In the Solver folder you'll find an example code o deploy any good model in production.

In the Result folder you'll find the results given by the best model trained using 100 epochs and many data.