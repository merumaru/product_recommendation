# product_recommendation
Recommendation engine based on images

## Setup

```
pip install -r requirements.txt
```

The server starts up at `<server-external-ip-address>:5000`

## Features

Recommend similar products by finding products with similar images.


### Add product to database

  To save 1024 dimensional feature vector for image of a product.

* **URL**

  /:productID/addImage
  Expects url of image in json payload with name of field being "url"

* **Method:**
  
  `POST`
  
* **Success Response:**
  
  * **Code:** 200 <br />

* **Error Response:**

  * **Code:** 400 <br />



### Recommend similar image

To return productIDs of similar images. 

- **URL**

  /:productID/similarProducts

- **Method:**

  `GET`

- **Success Response:**

  - **Code:** 200 <br />
    A JSON file of list of productIDs.

- **Error Response:**

  - **Code:** 400 <br />

----

