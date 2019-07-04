#!/usr/bin/csi -s
(import genann
        (only chicken.format printf)
        (only srfi-4 f64vector-ref))

(printf "Genann example 1.~n")
(printf "Train a small ANN to the XOR function using backpropagation.~n")
(define inputs #(#f64(0 0) #f64(0 1) #f64(1 0) #f64(1 1)))

(define outputs #(#f64(0) #f64(1) #f64(1) #f64(0)))

(define ann (make-genann 2 1 2 1))

(do ((i 0 (add1 i)))
    ((= i 300))
  (genann-train ann (vector-ref inputs 0) (vector-ref outputs 0) 3)
  (genann-train ann (vector-ref inputs 1) (vector-ref outputs 1) 3)
  (genann-train ann (vector-ref inputs 2) (vector-ref outputs 2) 3)
  (genann-train ann (vector-ref inputs 3) (vector-ref outputs 3) 3))


(do ((i 0 (add1 i)))
    ((= i 4))
  (printf "Output for ~a is ~a~n"
          (vector-ref inputs i)
          (inexact->exact
           (round
            (f64vector-ref (genann-run ann (vector-ref inputs i)) 0)))))
