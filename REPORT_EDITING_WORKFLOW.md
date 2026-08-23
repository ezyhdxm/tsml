# Report editing workflow

For the three queue / Poisson notes, the Markdown files are the source of truth:

- `poisson_process_for_order_book.md`
- `order_book_master_equation_derivation.md`
- `single_queue_stationary_and_first_hitting_times_guide.md`

Edit prose and equations in Markdown. Use ordinary LaTeX delimiters:

```markdown
Inline: $\lambda < \mu + \nu$

Display:
$$
E[T_1\mid V] = \frac{V}{\mu+\nu-\lambda}.
$$
```

The `.html` files are rendered artifacts. When iterating on the reports, edit the Markdown first and regenerate/update the corresponding HTML afterward. This keeps the prose easy to revise while allowing the HTML version to use robust offline math rendering.
