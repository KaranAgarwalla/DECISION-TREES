#lang racket

;candidate functions for the toy dataset
(provide y1)
(provide y2)
(provide y3)
(provide y4>62)

(define y1 (cons "feature1" (lambda (x) (car x)))) ; returns the value of feature 1 for a given test sample
(define y2 (cons "feature2" (lambda (x) (cadr x))))
(define y3 (cons "feature3" (lambda (x) (caddr x))))
(define y4>62 (cons "feature4>62" (lambda (x) (if (> (cadddr x) 62) 1 0)))) ; returns 1 if the value of feature 4 > 62, else 0

;candidate functions for the titanic dataset
(provide pclass)
(provide sex)
(provide age>25)
(provide sibsp)
(provide parch)
(provide fare>50)
(provide emb)

(define pclass (cons "pclass" (lambda (x) (car x)))) ; returns the value of pclass for a given test sample
(define sex (cons "sex" (lambda (x) (cadr x))))
(define age>25 (cons "age>25" (lambda (x) (if (> (caddr x) 25) 1 0))))
(define sibsp (cons "sibsp" (lambda (x) (cadddr x))))
(define parch (cons "parch" (lambda (x) (car (cddddr x)))))
(define fare>50 (cons "fare>50" (lambda (x) (if (> (cadr (cddddr x)) 50) 1 0))))
(define emb (cons "emb" (lambda (x) (caddr (cddddr x)))))

;candidate functions for the mushroom dataset
(provide cshape)
(provide csurf)
(provide bruise)
(provide odor)
(provide gatch)
(provide gspace)
(provide gsize)
(provide sshape)
(provide nring)
(provide pop)
(provide hab)

(define cshape (cons "cshape" (lambda (x) (car x))))
(define csurf (cons "csurf" (lambda (x) (cadr x))))
(define bruise (cons "bruise" (lambda (x) (caddr x))))
(define odor (cons "odor" (lambda (x) (cadddr x))))
(define gatch (cons "gatch" (lambda (x) (car (cddddr x)))))
(define gspace (cons "gspace" (lambda (x) (cadr (cddddr x)))))
(define gsize (cons "gsize" (lambda (x) (caddr (cddddr x)))))
(define sshape (cons "sshape" (lambda (x) (cadddr (cddddr x)))))
(define nring (cons "nring" (lambda (x) (car (cddddr (cddddr x))))))
(define pop (cons "pop" (lambda (x) (cadr (cddddr (cddddr x))))))
(define hab (cons "hab" (lambda (x) (last x))))
