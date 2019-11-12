from flask import Flask
app = Flask(__name__)

PRODUCT_ID_LIST = []
FATURES = []

@app.route("/<productID>/addImage", methods=['POST'])
def add_product(productID):
    return productID

@app.route("/<productID>/similarProducts", methods=['GET'])
def get_similar_product(productID):
    return productID

if __name__ == "__main__":
    app.run(host='0.0.0.0')