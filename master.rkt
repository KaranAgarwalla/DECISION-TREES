#lang racket

(require 2htdp/batch-io)

(require "decision_functions.rkt")

;input dataset
(provide toytrain)
(define toytrain "../data/toy_train.csv")

(provide titanictrain)
(define titanictrain "../data/titanic_train.csv")

(provide mushroomtrain)
(define mushroomtrain "../data/mushrooms_train.csv")

;output tree (dot file)
(provide toyout)
(define toyout "../output/toy-decision-tree.dot")

(define musout "../output/mushroom-decision-tree.dot")

;reading input datasets
;read the csv file myfile as a list of strings
;with each line of the original file as an element of the list
;further split each line at commas
;so then we have a list of list of strings
(provide toy-raw)
(define toy-raw (cdr (read-csv-file toytrain)))

(provide titanic-raw)
(define titanic-raw (map cddr (cdr (read-csv-file titanictrain))))
  
(provide mushroom-raw)
(define mushroom-raw (cdr (read-csv-file mushroomtrain)))

;function to convert data to internal numerical format
;(features . result)

(define (list-string-number list)
  (cond [(null? list) '()]
        [else (cons (string->number (car list)) (list-string-number (cdr list)))]))

(provide format)
(define (format data) (cons (list-string-number (cdr data)) (string->number (car data))))

;list of (features . result)
(provide toy)
(define toy (map format toy-raw))

(provide titanic)
(define titanic (map format titanic-raw))

(provide mushroom)
(define mushroom (map format mushroom-raw))

;============================================================================================================
;============================================================================================================
;============================================================================================================

;get fraction of result fields that are 1
;used to find probability value at leaf
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (sumofcdr l acc)
  (cond [(null? l) acc]
        [else (sumofcdr (cdr l) (+ acc (cdr (car l))))]))

(provide get-leaf-prob)
(define (get-leaf-prob data)
    (/ (sumofcdr data 0) (length data)))

(provide get-entropy)
(define (get-entropy data)
    (let ([true (get-leaf-prob data)])
      (- (+ (* true (logi true 2)) (* (- 1 true) (logi (- 1 true) 2))))))

(define (logi x b)
  (cond [(= x 0) 0]
        [else (log x b)]))

(define (sum l acc)
  (cond [(null? l) acc]
        [else (sum (cdr l) (+ acc (car l)))]))

;;find the difference in entropy achieved
;;by applying a decision function f to the data


(define (folding l acc n)
  (cond [(null? l) acc]
        [else (folding (cdr l) (+ acc (* (/ (length (car l)) n) (get-entropy (car l)))) n)]))

(provide entropy-diff)
;(define (entropy-diff f data)
;    (- (get-entropy data) (folding (group-by (lambda (x) (f (car x))) data) 0 (length data))))

(define (entropy-diff f data)
  (- (get-entropy data) (/ (sum (map (lambda (x) (* (length x) (get-entropy x))) (group-by (lambda (x) (f (car x))) data)) 0) (length data))))  
;;choose the decision function that most reduces entropy of the data
(provide choose-f)
(define (choose-f candidates data) ; returns a decision function
  (argmax (lambda (x) (entropy-diff (cdr x) data)) candidates))
;
(provide DTree)
(struct DTree (desc func kids) #:transparent)
;
;;build a decision tree (depth limited) from the candidate decision functions and data
(provide build-tree)

(define (build-tree candidates data depth)
  (cond [(= depth 0) (get-leaf-prob data)]
        [else (let ([best-f (choose-f candidates data)])
                (if (= depth 1) (DTree (car best-f) (cdr best-f) (map (lambda (x) (cons ((cdr best-f) (caar x)) (get-leaf-prob x)))
                                                                      (group-order (lambda (x) ((cdr best-f) (car x))) data)))
                    (DTree (car best-f) (cdr best-f) (map (lambda (x) (build-tree-helper ((cdr best-f) (caar x)) (remove best-f candidates) x (- depth 1)))
                                            (group-order (lambda (x) ((cdr best-f) (car x))) data)))))]))
                    

(define (build-tree-helper value candidates data depth)
  (cond [(null? data) (cons #t 0)]
        [(null? candidates) (cons value (get-leaf-prob data))]
        [(= depth 0) (cons value (get-leaf-prob data))]
        [else (let ([best-f (choose-f candidates data)])
                (if (= depth 1) (DTree (cons value (car best-f)) (cdr best-f) (map (lambda (x) (cons ((cdr best-f) (caar x)) (get-leaf-prob x)))
                                                                                   (group-order (lambda (x) ((cdr best-f) (car x))) data)))
                (DTree (cons value (car best-f)) (cdr best-f) (map (lambda (x) (build-tree-helper ((cdr best-f) (caar x)) (remove best-f candidates) x (- depth 1)))
                                                                   (group-order (lambda (x) ((cdr best-f) (car x))) data)))))]))

(define (group-order f data)
  (sort (group-by f data) (lambda (x y) (less-than (f (car x)) (f (car y))))))

(define (less-than x y)
  (cond [(equal? x #t) #f]
        [(equal? y #t) #t]
        [else (< x y)]))

;given a test data (features only), make a decision according to a decision tree
;returns probability of the test data being classified as 1
(provide make-decision)
(define (make-decision tree test)
  (cond [(pair? tree) (cdr tree)]
        [else (match tree [(DTree desc func kids) (make-decision (select-appropriate (func test) kids) test)])]))

(define (select-appropriate val kids)
  (cond [(null? kids) (cons "NO MATCHES" 0)]
        [else (let* ([elem (car kids)]
                 [value (if (pair? elem) (car elem) (car (DTree-desc elem)))])
                 (if (equal? value val) (car kids) (select-appropriate val (cdr kids))))]))

;
;;============================================================================================================
;;============================================================================================================
;;============================================================================================================
;;
;;annotate list with indices
;(define (pair-idx lst n)
;  (if (empty? lst) `() (cons (cons (car lst) n) (pair-idx (cdr lst) (+ n 1))))
;  )
;
;;generate tree edges (parent to child) and recurse to generate sub trees
;(define (dot-child children prefix tabs)
;  (apply string-append
;         (map (lambda (t)
;                (string-append tabs
;                               "r" prefix
;                               "--"
;                               "r" prefix "t" (~a (cdr t))
;                               "[label=\"" (~a (cdr t)) "\"];" "\n"
;                               (dot-helper (car t)
;                                           (string-append prefix "t" (~a (cdr t)))
;                                           (string-append tabs "\t")
;                                           )
;                               )
;                ) children
;                  )
;         )
;  )
;
;;generate tree nodes and call function to generate edges
;(define (dot-helper tree prefix tabs)
;  (let* ([node (match tree [(DTree d f c) (cons d c)])]
;         [d (car node)]
;         [c (cdr node)])
;    (string-append tabs
;                   "r"
;                   prefix
;                   "[label=\"" d "\"];" "\n\n"
;                   (dot-child (pair-idx c 0) prefix tabs)
;                   )
;    )
;  )
;
;;output tree (dot file)
;(provide display-tree)
;(define (display-tree tree outfile)
;  (write-file outfile (string-append "graph \"decision-tree\" {" "\n"
;                                     (dot-helper tree "" "\t")
;                                     "}"
;                                     )
;              )
;  )
;============================================================================================================
;============================================================================================================
;============================================================================================================
