(module genann (genann-init
                genann-copy
                genann-free!
                genann-train
                genann-randomize
                genann-read
                genann-write
                genann-run
                genann-inputs
                genann-hidden-layers
                genann-hidden-neurons
                genann-outputs
                genann-total-weights
                genann-weights

                ;; Added
                genann-init*
                make-genann
                genann-copy*
                genann?
                genann-weight-ref
                genann-weight-set!)

  (import scheme
          chicken.foreign
          chicken.fixnum
          (only chicken.file.posix
                port->fileno
                file-close
                fileno/stdin
                fileno/stdout)
          (only chicken.gc set-finalizer!)
          (only chicken.base
                unless
                define-record-type
                port?
                assert
                getter-with-setter)
          (only chicken.memory move-memory!)
          (only srfi-4 make-f64vector))

  (foreign-declare "#include \"genann.h\"")
  (foreign-declare "#include \"genann_src.c\"")

  (define-record-type genann
    (ptr->genann ptr)
    genann?
    (ptr genann->ptr))

  (define-foreign-type genann (c-pointer "genann") genann->ptr ptr->genann)

  (define genann-init
    (foreign-lambda genann "genann_init" int int int int))

  (define genann-copy
    (foreign-lambda genann "genann_copy" genann))

  (define genann-free!
    (foreign-lambda void "genann_free" genann))

  (define genann-train
    (foreign-lambda void "genann_train" genann f64vector f64vector double))

  (define genann-randomize
    (foreign-lambda void "genann_randomize" genann))

  (define %genann-read
    (foreign-lambda genann "genann_read" (c-pointer "FILE")))

  (define (extract-c-file port)
    (assert (port? port))
    (assert (eq? 'stream (##sys#slot port 7)))
    ((foreign-lambda* c-pointer ((scheme-object port))
       "C_return(C_block_item(port, 0));") port))

  (define (genann-read #!optional port)
    (%genann-read (extract-c-file port)))

  (define %genann-write
    (foreign-lambda void "genann_write" genann (c-pointer "FILE")))

  (define (genann-write ann #!optional port)
    (%genann-write ann (extract-c-file port)))

  (define %genann-run
    (foreign-lambda (c-pointer double) "genann_run" genann f64vector))

  (define (genann-run ann inputs)
    (let* ((len (genann-outputs ann))
           (out (make-f64vector len))
           (res (%genann-run ann inputs)))
      (move-memory! res
                    out
                    (* len (foreign-value "sizeof(double)" size_t)))
      out))

  (define genann-randomize
    (foreign-lambda void "genann_randomize" genann))

  ;; Added
  (define (genann-init* inputs hidden-layers hidden-neurons outputs)
    (set-finalizer! (genann-init inputs hidden-layers hidden-neurons outputs)
                    genann-free!))

  (define make-genann genann-init*)

  (define (genann-copy* genann)
    (set-finalizer! (genann-copy genann) genann-free!))

  (define genann-inputs
    (foreign-lambda* int ((genann ann))
      "C_return(ann->inputs);"))

  (define genann-hidden-layers
    (foreign-lambda* int ((genann ann))
      "C_return(ann->hidden_layers);"))

  (define genann-hidden-neurons
    (foreign-lambda* int ((genann ann))
      "C_return(ann->hidden);"))

  (define genann-outputs
    (foreign-lambda* int ((genann ann))
      "C_return(ann->outputs);"))

  (define genann-total-weights
    (foreign-lambda* int ((genann ann))
      "C_return(ann->total_weights);"))

  (define %genann-weights
    (foreign-lambda* (c-pointer double) ((genann ann))
      "C_return(ann->weight);"))

  (define (genann-weights ann)
    (let* ((len (genann-total-weights ann))
           (out (make-f64vector len))
           (res (%genann-weights ann)))
      (move-memory! res
                    out
                    (* len (foreign-value "sizeof(double)" size_t)))
      out))

  (define (genann-weight-set! ann i x)
    (assert (fx< i (genann-total-weights ann))
            'genann-weight-set!
            "out of range"
            (genann-weights ann)
            i)
    (foreign-lambda* void ((genann ann) (size_t i) (double x))
      "ann->weight[i] = x;"))

  (define genann-weight-ref
    (getter-with-setter
     (lambda (ann i)
       (assert (fx< i (genann-total-weights ann))
               'genann-weight-ref
               "out of range"
               (genann-weights ann)
               i)
       ((foreign-lambda* double ((genann ann) (size_t i))
          "C_return(ann->weight[i]);")
        ann i))
     genann-weight-set!)))

