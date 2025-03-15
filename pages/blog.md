---
layout: default
title: Blog
permalink: /blog/
---

<div class="post-container">
  {% for post in site.posts %} {% assign currDate = post.date | date: "%Y" %} {%
  if currDate != date %}
  <h1 class="archive-year">{{ currDate }}</h1>
  {% assign date = currDate %} {% endif %}
  <div class="archive-item">
    <a href="{{ post.url | relative_url }}" class="archive-title"
      >{{ post.title }}</a
    >
    <div class="archive-date"
      >{{ post.date | date: "%d %B %Y" }}</div
    >
    <div class='post-subtitle'
      >{{ post.subtitle }}
    </div>
  </div>
  {% endfor %}
</div>

