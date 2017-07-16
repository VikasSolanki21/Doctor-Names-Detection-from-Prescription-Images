# Doctor-Names-Detection-from-Prescription-Images


Aim: To detect doctor's name from prescription image.

Procedure: After fetching the prescription images from the urls present in the db, we rotated, cropped and adjusted the image so that tesseract is able to detect the doctor’s name present in the image. We also developed an algorithm which returns the doctor’s name from the text even if it is partially captured by the Tesseract. 
Also, ran Tesseract on client side, to obtain doctor names before the image is uploaded. 

Technologies used: PIL and OpenCV for image related operations, Tesseract OCR to detect text from the image

Result: List of possible doctor's names are returned with recall and precision being 48 % and 81 % respectively.
