from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from .models import Admin, Student
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   ##means from __init__.py import db

auth = Blueprint('auth', __name__)

@auth.route('/logout')
def logout():
    session.clear()
    session['status'] = 'out'
    return redirect(url_for('views.home'))

@auth.route('/studentLogin', methods=['GET', 'POST'])
def studentLogin():
    session['status'] = 'pending'
    session['position'] = 'student'

    if request.method == 'POST':
        srcode = request.form.get('srcode')
        password = request.form.get('password')

        user = Student.query.filter_by(studentSrcode=srcode).first()
        if user:
            if check_password_hash(user.studentPassword, password):
                flash('Logged in successfully!', category='success')
                session['status'] = 'in'
                userId = user.id
                session['userId'] = userId
                #session['userCollege'] = user.studentCollege
                #session['userClass'] = user.studentSection
                return redirect(url_for('views.chooseProf'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("studentLogin.html")


@auth.route('/studentSignUp', methods=['GET', 'POST'])
def studentSignUp():
    session['position'] = 'student'
    if request.method == 'POST':
        name = request.form.get('name')
        srcode = request.form.get('srcode')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        numCourse = request.form.get('numCourse')

        numCourse = int(numCourse)
        session['numCourse'] = numCourse

        user = Student.query.filter_by(studentSrcode=srcode).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(srcode) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 3:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = Student(studentName=name, studentSrcode=srcode, numOfCourses=numCourse, studentPassword=generate_password_hash(password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            flash('Account created!', category='success')
            session['status'] = 'addSubs'
            session['userId'] = new_user.id
            
            return redirect(url_for('views.addSubs'))

    return render_template("studentSignUp.html")


@auth.route('/adminLogin', methods=['GET', 'POST'])
def adminLogin():
    session['status'] = 'pending'
    session['position'] = 'admin'
    if request.method == 'POST':
        srcode = request.form.get('srcode')
        password = request.form.get('password')

        user = Admin.query.filter_by(facultySrcode=srcode).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                session['status'] = 'in'
                name = user.facultyName
                if name == "cj":
                    return redirect(url_for('views.showRec'))
                else:
                    return redirect(url_for('views.showRec'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("adminLogin.html")


@auth.route('/adminSignUp', methods=['GET', 'POST'])
def adminSignUp():
    session['status'] = 'pending'
    session['position'] = 'admin'
    if request.method == 'POST':
        srcode = request.form.get('srcode')
        name = request.form.get('name')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = Admin.query.filter_by(facultySrcode=srcode).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(srcode) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 3:
            flash('Password must be at least 3 characters.', category='error')
        else:
            new_user = Admin(facultySrcode=srcode, facultyName=name, password=generate_password_hash(
                password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            session['status'] = 'in'
            session['position'] = 'admin'
            flash('Account created!', category='success')
            return redirect(url_for('views.showRec'))

    return render_template("adminSignUp.html")