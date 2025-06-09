from flask import Flask, redirect, url_for, request, render_template, flash
import create_face
import svmcamera
import sklearn as sc
import joblib
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
index = -1
name = ""
#balance = joblib.load("balance")
naam = joblib.load("name.joblib")
menuitems ={'redbull': 100, 'cashew': 150, 'coke': 50, 'clips' : 80,'towel' : 80}

@app.route('/')
def index_page():
    # This will render the Homepage.html template
    # found in without/templates/Homepage.html
    return render_template('Homepage.html')

@app.route('/buy',methods = ['POST'])
def buy():
   dict = joblib.load("catalogue")
   res = []
   for a in dict:
      res.append(a)
   if request.method == 'POST':
     if request.form['ques'] == 'Yes':
      
      return render_template('products.html', menuitems=menuitems)
     elif request.form['ques'] == 'No':
      return render_template('Register.html',methods = ['POST','GET'])

@app.route('/products',methods = ['POST'])
def products():
  if request.method == 'POST' and request.form['buy']== 'Buy':
     names =[]
     prices = []
     value = request.form.getlist('add')
     count =0
     bill = ""
     for i in value:
        names.append(i)
        prices.append(menuitems[names[-1]])
        bill = bill + " " + i + " " + str(menuitems[i]) + "\n"
     print(value) 
     balance = joblib.load("balance")
     names = svmcamera.person()
     s=sum(prices)
     if (balance[names[1]]) - s >= 0:    
        balance[names[1]]= (balance[names[1]]) - s
        joblib.dump(balance, 'balance')
        return bill + "\n" + "Total bill is: " + str(s)
     else:
        flash("Do not have enough balance. Add " + str(s- balance[names[1]]))
        return render_template('balance.html',bal = balance[names[1]], person = names[0], method='POST')


@app.route('/Register',methods = ['POST', 'GET'])
def Register():
   if request.method == 'POST':
      user = request.form['nm']
      create_face.create_face_website(user)      
      #return redirect(url_for('buy',name = user))
      dict = joblib.load("catalogue")
      res = []
      for a in dict:
         res.append(a)
      balance = joblib.load("balance")
      balance.append(0)
      joblib.dump(balance,"balance")
      #names = svmcamera.person()
      #index = names.index(names[1])
      return render_template('Homepage.html')

@app.route('/Home',methods = ['POST', 'GET'])
def Home():
   if request.method == 'POST':
     if request.form['ques'] == 'Buy':
      names = svmcamera.person() 
      index = names[1]
      name = names[0]     
      return render_template('buy.html',person = names[0],methods = ['POST'])
     elif request.form['ques'] == 'Register':
      return render_template('Register.html',methods = ['POST','GET'])
     elif request.form['ques'] == 'Balance':
      names = svmcamera.person()#names = [name, index]; names[1] = index
      name = names[0]
      index = names[1]
      balance = joblib.load("balance")
      #return "Hello, Balance is: "+str(balance[names[1]]) + " and your name is: "+names[0]
      return render_template('balance.html',bal = balance[index], person = name)

@app.route('/Balance',methods = ['POST', 'GET'])
def Balance():
  if request.method == 'POST':
    if request.form['Button'] == 'Add':
      user = request.form['nm']
      money = int(user)
      #print(money)
      ls = svmcamera.person()
      
      
        
      name = ls[0]
      index = ls[1]
      print("Index is: "+name)
      balance = joblib.load("balance")
      balance[index] += money
      joblib.dump(balance,"balance")
      return "Success, balance added. Now it is: "+str(balance[index])
    elif request.form['Button'] == 'HomePage':
      return render_template('Homepage.html')

@app.route('/Train',methods = ['POST', 'GET'])
def Train():
  sc.svm_weight_create()
  a = joblib.load('name.joblib')
  return "Success on training. "+str(len(a))+" members."



if __name__ == '__main__':
   app.run(debug = True)