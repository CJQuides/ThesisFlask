from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Admin(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  facultyName = db.Column(db.String(255))
  facultySrcode = db.Column(db.String(255))
  password = db.Column(db.String(255))

class Student(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  studentName = db.Column(db.String(255))
  studentSrcode = db.Column(db.String(255))
  studentPassword = db.Column(db.String(255))
  numOfCourses = db.Column(db.Integer)
  studentsInfo = db.relationship('StudentsInfo')

class StudentsInfo(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  professorName = db.Column(db.String(255))
  enrolledCourse = db.Column(db.String(255))
  evaluationStatus = db.Column(db.String(255))
  studentSection = db.Column(db.String(255))
  studentCollege = db.Column(db.String(255))
  studentCampus = db.Column(db.String(255))
  student_id = db.Column(db.Integer, db.ForeignKey('student.id'))

class Result(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  facultyName = db.Column(db.String(255))
  campus = db.Column(db.String(255))
  college = db.Column(db.String(255))
  course = db.Column(db.String(255))
  facultyClass = db.Column(db.String(255))
  
  schoolYear = db.Column(db.String(255))
  semester = db.Column(db.String(255))
  
  studentsAnswered = db.Column(db.String(255))

  catA = db.Column(db.Float)
  catB = db.Column(db.Float)
  catC = db.Column(db.Float)
  catD = db.Column(db.Float)
  APS = db.Column(db.Float)
  sentimentTracker = db.Column(db.Float)
  senAnalysis = db.Column(db.String(255))
  comments = db.Column(db.String(255))
  result = db.Column(db.String(255))






"""""
INSERT INTO Result (facultyName, campus, college, course, facultyClass, studentsAnswered, catA, catB, catC, catD, APS, result) VALUES (LIPA,CABEIHM,ACC 101 - Financial Accounting and Reporting 1,BSMA-1101,24 / 44,4.66,4.57,4.67,4.64,4.63,Excellent);

INSERT INTO Result (campus, college, course, facultyClass, studentsAnswered, catA, catB, catC, catD, APS, result) VALUES ('LIPA','CABEIHM','ACC 101 - Financial Accounting and Reporting 1','BSMA-1101','24 / 44',4.66,'4.57','4.67','4.64','4.63','Excellent');

INSERT INTO Result (campus, college, course, facultyClass, studentsAnswered, catA, catB, catC, catD, APS, result) VALUES ('LIPA','CABEIHM','ACC 101 - Financial Accounting and Reporting 1','BSMA-1101','24 / 44',4.66,'4.57','4.67','4.64','4.63','Excellent');

class Note(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  data = db.Column(db.String(10000))
  rate = db.Column(db.Integer)
  date = db.Column(db.DateTime(timezone=True), default=func.now())
  user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class User(db.Model, UserMixin):
  id = db.Column(db.Integer, primary_key=True)
  email = db.Column(db.String(150), unique=True)
  password = db.Column(db.String(150))
  first_name = db.Column(db.String(150))
  notes = db.relationship('Note')
  #rating = db.relationship('Ratings')

class Ratings(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  rateA = db.Column(db.Integer)
  #user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Class(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  className = db.Column(db.String(255))
  classNumOfStudent = db.Column(db.Integer)
  facultyName = db.Column(db.String(255))
  course = db.Column(db.String(255))

ADMIN
INSERT INTO Admin (facultyName, facultySrcode, password) VALUES ('John', 'dapdap','123');
INSERT INTO Admin (facultyName, facultySrcode, password) VALUES ('Jane', 'dubidubi','123');
INSERT INTO Admin (facultyName, facultySrcode, password) VALUES ('Admin', 'admin','sha256$WIvwDWPhK9JDr7gl$6fe2173893e7270c0a0eb4615376e76dba403bff9f780543f6cdf3cea0723dc6');

CLASS
INSERT INTO Class (className, classNumOfStudent) VALUES ('CS1', 3);
INSERT INTO Class (className, classNumOfStudent) VALUES ('CS2', 2);
INSERT INTO Class (className, classNumOfStudent) VALUES ('CS3', 3);

STUDENT
1|CJ|19-32639|sha256$taEoUbwxkkUr3mwY$95fcbab3b6caeca1e9b5c85e9ef8463b7d14e52547085c0106f96abf1efb3d50||||
2|Cid|19-33242|sha256$gb8M5RTk5XpNXeXP$1281bf373583dd50ce6754565041f0333374380e176374173878069b892a0599||||
3|Viel|19-38262|sha256$qHgw1hzFjkBrO0I5$f07cabb18b855cb09aa674fdd87c12682da6bc2ba49be0643f82f6ca57e572d3||||


CREATE TABLE `classtbl` (
  `id` int(11) NOT NULL,
  `className` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `classNumOfStudent` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;



--

CREATE TABLE `facultyinfotbl` (
  `id` int(11) NOT NULL,
  `facultyName` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `course` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `section` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `campus` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `numOfStudents` int(11) DEFAULT NULL,
  `studentsAnswered` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;




CREATE TABLE `perfomance_evaluation` (
  `id` int(11) NOT NULL,
  `facultyName` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `campus` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `college` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `course` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `class` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `studentsAnswered` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `A` float NOT NULL,
  `B` float NOT NULL,
  `C` float NOT NULL,
  `D` float NOT NULL,
  `APS` float NOT NULL,
  `comments` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `result` varchar(255) COLLATE utf8_unicode_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--



CREATE TABLE `studentsinfotbl` (
  `id` int(11) NOT NULL,
  `name` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `srcode` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `enrolledCourse` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `section` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `professor` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `campus` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `evaluated` enum('NO','YES') COLLATE utf8_unicode_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;




CREATE TABLE `studenttbl` (
  `id` int(11) NOT NULL,
  `studentName` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `studentSrcode` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `studentPassword` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `studentClass` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `studentCollege` varchar(255) COLLATE utf8_unicode_ci NOT NULL,
  `studentCampus` varchar(255) COLLATE utf8_unicode_ci DEFAULT 'LIPA',
  `numOfCourses` int(11) DEFAULT 1
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

-- """