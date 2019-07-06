#!/usr/bin/csi -s
(import genann
        (only chicken.format printf)
        (only chicken.random pseudo-random-real)
        (only srfi-4 f64vector-ref)
        (only srfi-1 list-tabulate))

(printf "Genann example 1.~n")
(printf "Train a small ANN to the XOR function using random search.~n");

(define inputs #(#f64(0 0) #f64(0 1) #f64(1 0) #f64(1 1)))

(define outputs #(#f64(0) #f64(1) #f64(1) #f64(0)))

(let loop ((ann (make-genann 2 1 2 1))
           (last-err 1000.0)
           (count 1))
  (let ((save (genann-copy* ann)))
    (when (= 0 (modulo count 1000))
      (genann-randomize ann)
      (set! last-err 1000.0))
    (do ((i 0 (add1 i)))
        ((= i (genann-total-weights ann)))
      (set! (genann-weight-ref ann i) (+ (genann-weight-ref ann i)
                                         (- (pseudo-random-real) .5))))
    (let ((err
           (apply + (list-tabulate
                         4
                         (lambda (i)
                           (expt (- (f64vector-ref
                                     (genann-run ann (vector-ref inputs i)) 0)
                                    (f64vector-ref (vector-ref outputs i) 0))
                                 2.0))))))

      (cond ((<= err 0.01)
             (printf "Finished in ~a loops~n" count)
             (do ((i 0 (add1 i)))
                 ((= i 4))
               (printf "Output for ~a is ~a~n"
                       (vector-ref inputs i)
                       (inexact->exact
                        (round
                         (f64vector-ref (genann-run ann (vector-ref inputs i)) 0))))))
            ((< err last-err)
             (loop ann err (add1 count)))
            (else
             (loop save last-err (add1 count)))))))


