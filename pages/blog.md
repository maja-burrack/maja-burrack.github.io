---
layout: default
title: Blog
permalink: /blog/
---

<div class="custom-container">
  {% for post in site.posts %} {% assign currDate = post.date | date: "%Y" %} {%
  if currDate != date %}
  <h1 class="archive-year">{{ currDate }}</h1>
  {% assign date = currDate %} {% endif %}
  <div class="archive-item">
    <span class="post-date archive-date"
      >{{ post.date | date: "%B %-d, %Y" }}</span
    >
    <a href="{{ post.url | relative_url }}" class="archive-title"
      >{{ post.title }}</a
    >
  </div>
  {% endfor %}
</div>

