(module genann (genann-init
                genann-copy
                genann-free!
                genann-train
                genann-read
                genann-write
                genann-run
                ;; Added
                make-genann
                genann?
                genann-inputs
                genann-hidden-layers
                genann-hidden-neurons
                genann-outputs)

  (import scheme
          chicken.foreign
          (only chicken.file.posix
                port->fileno
                file-close
                fileno/stdin
                fileno/stdout)
          (only chicken.gc set-finalizer!)
          (only chicken.base unless define-record-type port? assert)
          (only chicken.memory move-memory!)
          (only srfi-4 make-f64vector))

  (foreign-declare "#include \"genann.h\"")

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

  (define fdopen
    (foreign-lambda (c-pointer "FILE") "fdopen" int c-string))

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
    (let* ((len (genann-outputs (genann->ptr ann)))
           (out (make-f64vector len))
           (res (%genann-run ann inputs)))
      (move-memory! res
                    out
                    (* len (foreign-value "sizeof(double)" size_t)))
      out))

  (define (make-genann inputs hidden-layers hidden-neurons outputs)
    (set-finalizer! (genann-init inputs hidden-layers hidden-neurons outputs)
                    genann-free!))

  (define genann-inputs
    (foreign-lambda* int (((c-pointer "genann") ann))
      "C_return(ann->inputs);"))

  (define genann-hidden-layers
    (foreign-lambda* int (((c-pointer "genann") ann))
      "C_return(ann->hidden_layers);"))

  (define genann-hidden-neurons
    (foreign-lambda* int (((c-pointer "genann") ann))
      "C_return(ann->hidden);"))

  (define genann-outputs
    (foreign-lambda* int (((c-pointer "genann") ann))
      "C_return(ann->outputs);")))
