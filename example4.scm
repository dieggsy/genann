#!/usr/bin/csi -s
(import genann
        (only chicken.io read-line)
        (only chicken.string string-split)
        (only chicken.format printf)
        (only srfi-4 f64vector-ref list->f64vector)
        (only srfi-1 split-at))

(printf "Genann example 4.~n")
(printf "Train an ANN on the IRIS dataset using backpropagation.~n")

(define iris-data "example/iris.data")

(define class-names '(("Iris-setosa" . #f64(1 0 0))
                      ("Iris-versicolor" . #f64(0 1 0))
                      ("Iris-virginica" . #f64(0 0 1))))

(define (load-data)
 (call-with-input-file iris-data
   (lambda (p)
     (let loop ((line (read-line p))
                (samples 0)
                (inputs '())
                (class '()))
       (if (eof-object? line)
           (values samples (list->vector inputs) (list->vector class))
           (let*-values (((split) (string-split line ","))
                         ((ilist clist) (split-at split 4)))
             (loop
              (read-line p)
              (add1 samples)
              (cons (list->f64vector (map string->number ilist))
                    inputs)
              (cons (alist-ref (car clist) class-names string=?) class))))))))

(define-values (samples input class) (load-data))

(printf "Loaded ~a data points from ~a~n" samples iris-data)

(define ann (make-genann 4 1 4 3))

(define loops 5000)

(printf "Training for ~a loops over data.\n" loops)

(do ((i 0 (add1 i)))
    ((= i loops))
  (do ((j 0 (add1 j)))
      ((= j samples))
    (genann-train ann (vector-ref input j) (vector-ref class j) .01)))

(define ~ f64vector-ref)


(let loop ((j 0)
           (correct 0))
  (if (= j samples)
      (printf "~a/~a correct (~a%)." correct samples
              (round (* 100 (/ correct samples 1.0))))
      (let ((guess (genann-run ann (vector-ref input j))))
        (cond ((= 1.0 (~ (vector-ref class j) 0))
               (if (and (> (~ guess 0) (~ guess 1))
                        (> (~ guess 0) (~ guess 2)))
                   (loop (add1 j) (add1 correct))
                   (loop (add1 j) correct)))
              ((= 1.0 (~ (vector-ref class j) 1))
               (if (and (> (~ guess 1) (~ guess 0))
                        (> (~ guess 1) (~ guess 2)))
                   (loop (add1 j) (add1 correct))
                   (loop (add1 j) correct)))
              ((= 1.0 (~ (vector-ref class j) 2))
               (if (and (> (~ guess 2) (~ guess 0))
                        (> (~ guess 2) (~ guess 1)))
                   (loop (add1 j) (add1 correct))
                   (loop (add1 j) correct)))
              (else
               (printf "Logic error.~n")
               (exit 1))))))
