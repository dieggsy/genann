#!/usr/bin/csi -s
(import genann
        (only chicken.format printf)
        (only srfi-4 f64vector-ref))

(printf "Genann example 1.~n")
(printf "Train a small ANN to the XOR function using backpropagation.~n")

;; Input and expected out data for the XOR function.
(define inputs #(#f64(0 0) #f64(0 1) #f64(1 0) #f64(1 1)))
(define outputs #(#f64(0) #f64(1) #f64(1) #f64(0)))

;; New network with 2 inputs, 1 hidden layer of 2 neurons, and 1 output.
(define ann (make-genann 2 1 2 1))

;; Train on the four labeled data points many times.
(do ((i 0 (add1 i)))
    ((= i 300))
  (genann-train ann (vector-ref inputs 0) (vector-ref outputs 0) 3)
  (genann-train ann (vector-ref inputs 1) (vector-ref outputs 1) 3)
  (genann-train ann (vector-ref inputs 2) (vector-ref outputs 2) 3)
  (genann-train ann (vector-ref inputs 3) (vector-ref outputs 3) 3))

;; Run the network and see what it predicts.
(do ((i 0 (add1 i)))
    ((= i 4))
  (printf "Output for ~a is ~a~n"
          (vector-ref inputs i)
          (inexact->exact
           (round
            (f64vector-ref (genann-run ann (vector-ref inputs i)) 0)))))
