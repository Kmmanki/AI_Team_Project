from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def indexPage():
    return '첫 화면'

@app.route("/")
def hello():
    return 'Hello Wrold'

@app.route('/htmlPage')
def htmlPage():
    return render_template('graph.html')

from flask import request, redirect, url_for

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'error'
        else:
            error = 'success'
    return render_template('login.html', error=error)

@app.route('/piano')
def wowow():
    return render_template('piano.html')



if __name__ ==  "__main__":
    app.run()