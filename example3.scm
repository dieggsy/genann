#!/usr/bin/csi -s
(import genann
        (only chicken.format printf)
        (only srfi-4 f64vector-ref))

(define save-name "example/xor.ann")

(printf "Genann example 3.~n")
(printf "Load a saved ANN to solve the XOR function.~n")

(define ann (call-with-input-file save-name
              (cut genann-read <>)))

(define inputs #(#f64(0 0) #f64(0 1) #f64(1 0) #f64(1 1)))

(do ((i 0 (add1 i)))
    ((= i 4))
  (printf "Output for ~a is ~a~n"
          (vector-ref inputs i)
          (inexact->exact
           (round
            (f64vector-ref (genann-run ann (vector-ref inputs i)) 0)))))
